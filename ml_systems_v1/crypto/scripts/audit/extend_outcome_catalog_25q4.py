"""extend_outcome_catalog_25q4.py - Extend outcome_catalog.parquet through 2025-12-31.

Adds Q4 2025 rows (2025-10-01 to 2025-12-31, ~92 days) to the existing catalog
which ends 2025-09-30. Stops at 2025-12-31; does NOT touch UNSEEN (2026-01-01+).

Per current §7b convention:
  TRAIN  = 2020-2024
  VAL    = 2025-Q1-Q3 (Jan-Sep 2025)
  OOS    = 2025-Q4   (Oct-Dec 2025)    <-- this script adds
  UNSEEN = 2026-Q1-Apr                  <-- NEVER touched

Uses same K=5 LO oracle methodology as scripts/research/build_outcome_catalog.py.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import date

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CHIMERA_1D = ROOT / "data/processed/chimera/1d"
OUT_CATALOG = ROOT / "data/processed/outcome_catalog.parquet"

# Hard boundary: do NOT touch 2026+ UNSEEN
EXTEND_START = date(2025, 10, 1)
EXTEND_END = date(2025, 12, 31)

K = 5
TAKER_COST_RT = 0.0024

HIGH_THRESH = 0.02
MED_THRESH = 0.005
LOW_THRESH = -0.005


def get_asset_chimera_map():
    mapping = {}
    for f in os.listdir(CHIMERA_1D):
        if not f.endswith(".parquet"):
            continue
        sym_raw = f.split("usdt")[0].upper()
        path = CHIMERA_1D / f
        if sym_raw not in mapping or str(path) > str(mapping[sym_raw]):
            mapping[sym_raw] = path
    return mapping


def load_all_closes(asset_map):
    frames = []
    for sym, path in sorted(asset_map.items()):
        df = pl.read_parquet(path, columns=["date", "close"])
        df = df.rename({"close": sym})
        frames.append(df)
    wide = frames[0]
    for df in frames[1:]:
        wide = wide.join(df, on="date", how="full", coalesce=True)
    wide = wide.sort("date")
    return wide


def compute_oracle_rows(wide_df, start_date, end_date, k=5):
    assets = [c for c in wide_df.columns if c != "date"]
    n = len(wide_df)
    dates = wide_df["date"].to_list()
    close_arr = wide_df.select(assets).to_numpy().astype(float)

    rows = []
    for i in range(n):
        d = dates[i]
        if d < start_date or d > end_date:
            continue
        if i + 5 >= n:
            continue

        rets_1d, rets_3d, rets_5d = {}, {}, {}
        for j, sym in enumerate(assets):
            c0 = close_arr[i, j]
            if c0 > 0 and not np.isnan(c0):
                c1 = close_arr[i + 1, j]
                c3 = close_arr[i + 3, j]
                c5 = close_arr[i + 5, j]
                if c1 > 0 and not np.isnan(c1): rets_1d[sym] = c1 / c0 - 1
                if c3 > 0 and not np.isnan(c3): rets_3d[sym] = c3 / c0 - 1
                if c5 > 0 and not np.isnan(c5): rets_5d[sym] = c5 / c0 - 1

        def oracle_k5(rets_dict):
            if len(rets_dict) < k:
                return None, None
            srt = sorted(rets_dict.items(), key=lambda x: x[1], reverse=True)
            picks = srt[:k]
            avg_ret = float(np.mean([r for _, r in picks]))
            return avg_ret - TAKER_COST_RT, [s for s, _ in picks]

        net_1d, picks_1d = oracle_k5(rets_1d)
        net_3d, picks_3d = oracle_k5(rets_3d)
        net_5d, picks_5d = oracle_k5(rets_5d)
        if net_1d is None:
            continue

        def classify(r):
            if r >= HIGH_THRESH: return "HIGH"
            if r >= MED_THRESH:  return "MED"
            if r >= LOW_THRESH:  return "LOW"
            return "NEG"

        rows.append({
            "date": d,
            "ideal_k5_1d_ret": float(net_1d),
            "ideal_k5_3d_ret": float(net_3d) if net_3d is not None else None,
            "ideal_k5_5d_ret": float(net_5d) if net_5d is not None else None,
            "day_class_1d": classify(net_1d),
            "day_class_3d": classify(net_3d) if net_3d is not None else None,
            "day_class_5d": classify(net_5d) if net_5d is not None else None,
            "winning_picks_1d": str(picks_1d) if picks_1d else None,
            "winning_picks_3d": str(picks_3d) if picks_3d else None,
            "winning_picks_5d": str(picks_5d) if picks_5d else None,
            "n_assets_available_1d": len(rets_1d),
            "gross_k5_1d_ret": float(net_1d + TAKER_COST_RT),
        })
    return pl.DataFrame(rows)


def main():
    print(f"[extend] loading 87 chimera 1d assets ...")
    asset_map = get_asset_chimera_map()
    print(f"[extend] mapped {len(asset_map)} assets")
    wide = load_all_closes(asset_map)
    print(f"[extend] wide close panel rows={len(wide)} date_range={wide['date'].min()} -> {wide['date'].max()}")

    print(f"[extend] computing 25Q4 oracle rows ({EXTEND_START} -> {EXTEND_END}) ...")
    new_rows = compute_oracle_rows(wide, EXTEND_START, EXTEND_END)
    print(f"[extend] new rows: {len(new_rows)}")

    # Add day_class_q if it's in the existing schema (placeholder; recomputed properly in full builder)
    if "day_class_q" not in new_rows.columns:
        new_rows = new_rows.with_columns(pl.lit("MID_Q").alias("day_class_q"))

    # Load existing and append
    existing = pl.read_parquet(str(OUT_CATALOG))
    print(f"[extend] existing rows: {len(existing)}  ({existing['date'].min()} -> {existing['date'].max()})")

    # Reorder new_rows columns to match existing
    new_rows = new_rows.select(existing.columns)

    combined = pl.concat([existing, new_rows]).sort("date").unique(subset=["date"], keep="first")
    print(f"[extend] combined rows: {len(combined)}  ({combined['date'].min()} -> {combined['date'].max()})")

    # Verify no UNSEEN burn
    n_unseen = (combined["date"] >= date(2026, 1, 1)).sum()
    if n_unseen > 0:
        raise RuntimeError(f"REFUSING TO WRITE: {n_unseen} rows in UNSEEN territory (2026+)")
    print(f"[extend] UNSEEN check: 0 rows in 2026+ -> safe")

    # Backup existing first
    backup = OUT_CATALOG.with_suffix(".parquet.pre25q4_backup")
    if not backup.exists():
        existing.write_parquet(str(backup))
        print(f"[extend] backed up existing to {backup}")

    combined.write_parquet(str(OUT_CATALOG))
    print(f"[extend] wrote {OUT_CATALOG} rows={len(combined)}")

    # Summary for the new region
    new_region = combined.filter((pl.col("date") >= EXTEND_START) & (pl.col("date") <= EXTEND_END))
    print(f"\n[extend] 25Q4 stats:")
    print(f"  rows: {len(new_region)}")
    print(f"  ideal_k5_1d_ret: mean={new_region['ideal_k5_1d_ret'].mean():.4f}  median={new_region['ideal_k5_1d_ret'].median():.4f}")
    print(f"  pct days >= +2%: {((new_region['ideal_k5_1d_ret'] >= 0.02).sum() / len(new_region) * 100):.1f}%")


if __name__ == "__main__":
    main()
