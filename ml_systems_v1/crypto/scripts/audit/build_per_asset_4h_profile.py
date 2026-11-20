"""build_per_asset_4h_profile.py -- 4h-cadence per-asset MA/EMA profile.

Companion to build_per_asset_ma_ema_profile.py (which uses 1d cadence).
Filters pair_by_asset_cadence to cadence == '4h' to test sub-day deployment.

Output: data/processed/per_asset_ma_ema_profile_4h.parquet
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PER_ASSET_DIR = ROOT / "runs" / "oracle_layer3" / "ma_ema_per_asset_train"
PERMUT_DIR = ROOT / "runs" / "oracle_layer3" / "ma_ema_permutation"
OWN_REGIME_PATH = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
OUT_PARQUET = ROOT / "data" / "processed" / "per_asset_ma_ema_profile_4h.parquet"

# Same gates as 1d build
MIN_N_SIGNALED = 20
MIN_HIT_RATE = 0.45
MIN_MEAN_NET_PCT = 0.10
TOP_K_PER_ASSET = 10
N_REGIME_MIN = 3
SHARPE_SURVIVE = 0.05
SHARPE_BREAK = -0.15


def survival_label(sharpe, n):
    if n < N_REGIME_MIN: return "insufficient"
    if sharpe > SHARPE_SURVIVE: return "survives"
    if sharpe < SHARPE_BREAK: return "breaks"
    return "neutral"


def classify_cell(per_own):
    labels = {r: survival_label(s["sharpe"] if s else 0, s["n"] if s else 0) for r, s in per_own.items()}
    survives = {r for r, l in labels.items() if l == "survives"}
    breaks_ = {r for r, l in labels.items() if l == "breaks"}
    neutral = {r for r, l in labels.items() if l == "neutral"}
    if len({"bull", "chop", "bear", "crash"} & survives) >= 3: return "ALL_WEATHER"
    if {"bull", "chop", "bear"} <= (survives | neutral) and "crash" in breaks_: return "BLOCK_OWN_CRASH"
    if {"bull", "chop"} <= (survives | neutral) and ({"bear", "crash"} & breaks_): return "BLOCK_OWN_BEAR"
    if {"bull", "chop"} <= survives: return "BULL_AND_CHOP"
    if "bull" in survives and not ({"chop", "bear", "crash"} & survives): return "BULL_ONLY"
    return "REGIME_DEPENDENT"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("PER-ASSET 4h MA/EMA PROFILE BUILD")
    print("="*78)
    per_asset = pl.read_parquet(PER_ASSET_DIR / "pair_by_asset_cadence.parquet").to_pandas()
    print(f"Total per_asset_cadence rows: {len(per_asset):,}")

    per_asset_4h = per_asset[per_asset["cadence"] == "4h"].copy()
    print(f"4h rows: {len(per_asset_4h):,} across {per_asset_4h['asset'].nunique()} assets")

    # Load asset metadata
    snap = pl.read_parquet(PERMUT_DIR / "event_ma_snapshot.parquet").to_pandas()
    snap["date"] = pd.to_datetime(snap["date"])
    asset_to_bucket = {a: g["bucket"].iloc[0] for a, g in snap.groupby("asset")}

    # Per asset: take top-10 by sharpe_proxy filtered by gates
    profile_rows = []
    for asset, sub in per_asset_4h.groupby("asset"):
        if asset not in asset_to_bucket: continue
        bucket = asset_to_bucket[asset]
        qualified = sub[(sub["n_signaled"] >= MIN_N_SIGNALED) &
                         (~sub["degenerate_signal"]) &
                         (~sub["signal_quasi_constant"]) &
                         (sub["hit_rate"] >= MIN_HIT_RATE) &
                         (sub["mean_pnl_pct"] >= MIN_MEAN_NET_PCT)]
        if qualified.empty: continue
        top = qualified.nlargest(TOP_K_PER_ASSET, "sharpe_proxy")
        for _, c in top.iterrows():
            # No event_ma_snapshot for 4h — we don't have per-event MA values at 4h cadence
            # Use overall stats as proxy; regime tagging is COARSE (cannot break out per regime)
            # Mark all cells as REGIME_DEPENDENT (conservative — bull+chop only)
            profile_rows.append({
                "cell_id": f"{asset}|{c['ma_type']}|{c['fast']}|{c['slow']}",
                "asset": asset, "bucket": bucket, "cadence": "4h",
                "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                "overall_n": int(c["n_signaled"]),
                "overall_mean_pct": float(c["mean_pnl_pct"]),
                "overall_hit": float(c["hit_rate"]),
                "sharpe": float(c["sharpe_proxy"]),
                "regime_tag": "REGIME_DEPENDENT",  # bull+chop (conservative — no per-regime breakdown at 4h)
                "is_cousin_set_member": False,
            })

    if not profile_rows:
        print("[FATAL] no rows generated")
        return 2

    profile_df = pd.DataFrame(profile_rows)

    # Cousin selection: top-3 per asset by sharpe (no signal-correlation pruning at 4h —
    # we lack per-event MA snapshots; just take top-3)
    profile_df["sharpe_rank"] = profile_df.groupby("asset")["sharpe"].rank(method="first", ascending=False)
    profile_df["is_cousin_set_member"] = profile_df["sharpe_rank"] <= 3
    profile_df = profile_df.drop(columns=["sharpe_rank"])

    print(f"\nProfile rows: {len(profile_df)}; assets: {profile_df['asset'].nunique()}")
    print(f"Cousin members: {profile_df['is_cousin_set_member'].sum()}")
    print(f"Median sharpe (cousins): {profile_df[profile_df['is_cousin_set_member']]['sharpe'].median():+.4f}")
    print(f"Median mean per-event (cousins): {profile_df[profile_df['is_cousin_set_member']]['overall_mean_pct'].median():+.3f}%")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[OK] wrote {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
