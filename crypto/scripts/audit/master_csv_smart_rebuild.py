"""master_csv_smart_rebuild.py -- master CSV rebuild + V8 sim (smart-grid + V6).

INPUTS:
  - Chimera 1d/4h/1h per asset
  - SmartCandidateGenerator: Fibonacci + golden + log-spaced + decorrelation
  - TRAIN+VAL/OOS split: 2024-05-15 boundary

OUTPUTS:
  - master_smart_cells_per_asset_cadence.parquet -- full cell library
        cols: asset, cadence, ma_type, fast, slow, n_train, sharpe_train,
              hit_train, mean_pnl_train, n_oos, sharpe_oos, hit_oos, mean_pnl_oos,
              pct_bars_active, regime_pct_active_*, bucket
  - MASTER_TOP_PER_ASSET_CADENCE_SMART_2026_05_20.csv -- top cell per
        (asset, cadence), refreshed master CSV
  - MASTER_PER_ASSET_STORY_2026_05_20.md -- per-asset narrative + recommendation
  - per_asset_smart_grid_profile.parquet -- subset for V8 sim
  - V8_OOS_RESULTS.json -- V6 architecture with smart-grid cells

DESIGN:
  - Per asset+cadence: generate ~65 smart candidates (Fib/golden/log) then
    empirically decorrelate at threshold 0.85 on TRAIN+VAL closes.
  - For each surviving cell: compute cross-up events on TRAIN+VAL + OOS;
    derive Sharpe-proxy (mean / std of forward 5-bar return), hit rate,
    mean PnL, state-based stats (% bars active).
  - Tag with bucket, sector, regime-conditional active%.
  - Top cell per (asset, cadence) = highest TRAIN+VAL Sharpe-proxy with
    n_train >= 10 fires.
  - V8 sim: use smart-grid cells as the profile, run V6 architecture
    (confirmation gate + 3-level ranker on 1d/4h/1h MA-active states).

INVARIANTS:
  - TRAIN+VAL only for cell selection; OOS reserved for evaluation
  - No look-ahead in stats (forward returns at training time use training closes)
  - Long-only, spot, no leverage
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from smart_candidate_generator import generate_raw_candidates, empirical_decorrelate

OUT_DIR = ROOT / "runs" / "audit" / "MASTER_CSV_SMART_REBUILD_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_VAL_END = date(2024, 5, 15)
OOS_START = date(2024, 5, 16)
OOS_END = date(2025, 3, 15)
MIN_FIRES_TRAIN = 10
MIN_FIRES_OOS_REPORT = 1
FWD_BARS_TRAIN = 5
FWD_BARS_OOS = 5
CORR_THRESHOLD = 0.85

U50 = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC",
    "SUI", "NEAR", "APT", "DOT", "HBAR", "ALGO", "AR",
    "PEPE", "SHIB", "FLOKI", "BONK",
    "UNI", "AAVE", "JST", "CRV", "FET",
    "ICP", "FIL", "ETC", "TRX", "XLM", "BCH", "OP", "ARB", "CHZ",
    "ENJ", "ZEC", "DASH", "LDO", "SUPER", "DYDX",
]
CADENCES = ["1d", "4h", "1h"]


def _ma(closes: np.ndarray, period: int, ma_type: str) -> np.ndarray:
    sr = pd.Series(closes)
    if ma_type == "SMA":
        return sr.rolling(period).mean().values
    return sr.ewm(span=period, adjust=False).mean().values


def _cross_up(closes: np.ndarray, fast: int, slow: int, ma_type: str) -> np.ndarray:
    if len(closes) < slow + 2:
        return np.zeros(len(closes), dtype=bool)
    mf = _ma(closes, fast, ma_type)
    ml = _ma(closes, slow, ma_type)
    cross = np.zeros(len(closes), dtype=bool)
    cross[1:] = (mf[1:] > ml[1:]) & (mf[:-1] <= ml[:-1])
    return cross


def _ma_active(closes: np.ndarray, fast: int, slow: int, ma_type: str) -> np.ndarray:
    if len(closes) < slow + 2:
        return np.zeros(len(closes), dtype=bool)
    mf = _ma(closes, fast, ma_type)
    ml = _ma(closes, slow, ma_type)
    return mf > ml


def _evaluate_cell(closes: np.ndarray, dates: np.ndarray, fast: int, slow: int,
                   ma_type: str, train_end: date, oos_start: date, oos_end: date,
                   fwd_train: int, fwd_oos: int) -> dict:
    cross = _cross_up(closes, fast, slow, ma_type)
    active = _ma_active(closes, fast, slow, ma_type)
    n = len(closes)
    train_pnl = []
    oos_pnl = []
    for i in range(n - fwd_oos - 1):
        if not cross[i]:
            continue
        d = dates[i]
        ep = closes[i]
        if ep <= 0 or not np.isfinite(ep):
            continue
        if d <= train_end:
            fp_i = i + fwd_train
            if fp_i >= n:
                continue
            fp = closes[fp_i]
            if not np.isfinite(fp) or fp <= 0:
                continue
            train_pnl.append(fp / ep - 1)
        elif oos_start <= d <= oos_end:
            fp_i = i + fwd_oos
            if fp_i >= n:
                continue
            fp = closes[fp_i]
            if not np.isfinite(fp) or fp <= 0:
                continue
            oos_pnl.append(fp / ep - 1)
    train_arr = np.array(train_pnl) if train_pnl else np.array([])
    oos_arr = np.array(oos_pnl) if oos_pnl else np.array([])
    out = {
        "fast": fast, "slow": slow, "ma_type": ma_type,
        "n_train": int(len(train_arr)),
        "mean_pnl_train": float(train_arr.mean()) if len(train_arr) else 0.0,
        "hit_train": float((train_arr > 0).mean()) if len(train_arr) else 0.0,
        "sharpe_train": float(train_arr.mean() / train_arr.std()) if len(train_arr) > 1 and train_arr.std() > 0 else 0.0,
        "max_pnl_train": float(train_arr.max()) if len(train_arr) else 0.0,
        "min_pnl_train": float(train_arr.min()) if len(train_arr) else 0.0,
        "n_oos": int(len(oos_arr)),
        "mean_pnl_oos": float(oos_arr.mean()) if len(oos_arr) else 0.0,
        "hit_oos": float((oos_arr > 0).mean()) if len(oos_arr) else 0.0,
        "sharpe_oos": float(oos_arr.mean() / oos_arr.std()) if len(oos_arr) > 1 and oos_arr.std() > 0 else 0.0,
        "max_pnl_oos": float(oos_arr.max()) if len(oos_arr) else 0.0,
        "min_pnl_oos": float(oos_arr.min()) if len(oos_arr) else 0.0,
        "pct_bars_active_train": float(active[:len(active)][np.array([d <= train_end for d in dates[:len(active)]])].mean()) if len(active) else 0.0,
    }
    return out


def mine_per_asset_cadence(chimera_loader, asset: str, cadence: str,
                            raw_cands: list[tuple[int, int]]) -> list[dict]:
    """Decorrelate candidates on asset's TRAIN closes, then evaluate each."""
    try:
        df = chimera_loader.load(asset, cadence)
    except Exception:
        return []
    if df is None:
        return []
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("timestamp" if "timestamp" in df.columns else "date").reset_index(drop=True)
    closes = df["close"].values.astype(float)
    dates = df["date"].values
    if len(closes) < 200:
        return []
    train_closes = closes[np.array([d < TRAIN_VAL_END for d in dates])]
    rows = []
    for ma_type in ["SMA", "EMA"]:
        if len(train_closes) >= 200:
            cands = empirical_decorrelate(train_closes, raw_cands, CORR_THRESHOLD, ma_type)
        else:
            cands = raw_cands
        for (fast, slow) in cands:
            r = _evaluate_cell(closes, dates, fast, slow, ma_type,
                                TRAIN_VAL_END, OOS_START, OOS_END,
                                FWD_BARS_TRAIN, FWD_BARS_OOS)
            r.update({"asset": asset, "cadence": cadence})
            rows.append(r)
    return rows


def regime_conditional_active(closes_1d: np.ndarray, dates_1d: np.ndarray,
                              fast: int, slow: int, ma_type: str,
                              own_regime_for_asset: pd.DataFrame, train_end: date) -> dict:
    """Compute % bars active conditional on each regime (bull/chop/bear/crash)."""
    active = _ma_active(closes_1d, fast, slow, ma_type)
    n = min(len(active), len(dates_1d))
    if not n:
        return {"bull": 0, "chop": 0, "bear": 0, "crash": 0}
    df = pd.DataFrame({"date": dates_1d[:n], "active": active[:n]})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < pd.Timestamp(train_end)]
    if not len(df):
        return {"bull": 0, "chop": 0, "bear": 0, "crash": 0}
    own = own_regime_for_asset.copy()
    own["date"] = pd.to_datetime(own["date"])
    merged = df.merge(own[["date", "asset_own_regime"]], on="date", how="inner")
    out = {}
    for r in ["bull", "chop", "bear", "crash"]:
        sub = merged[merged["asset_own_regime"] == r]
        out[r] = float(sub["active"].mean()) if len(sub) else 0.0
    return out


def main():
    from pipeline.chimera_loader import ChimeraLoader
    cl = ChimeraLoader()

    print("=" * 78)
    print("MASTER CSV SMART REBUILD + V8 (V6 + smart-grid cells)")
    print(f"  TRAIN+VAL end: {TRAIN_VAL_END}  OOS: {OOS_START} -> {OOS_END}")
    print(f"  universe: {len(U50)} assets  cadences: {CADENCES}")
    print("=" * 78)

    raw_cands = generate_raw_candidates(max_period=100)
    print(f"\nraw smart candidates: {len(raw_cands)}")

    # ---- universe bucket / sector map
    import yaml
    asset_meta = {}
    for p in (ROOT/"config"/"universes"/"u50.yaml", ROOT/"config"/"universes"/"u100.yaml"):
        with open(p) as f:
            doc = yaml.safe_load(f)
        for a in doc.get("assets", []) + doc.get("extra_assets", []):
            if a.get("status", "ready") != "ready":
                continue
            sym = a["symbol"].replace("USDT", "")
            asset_meta[sym] = {"bucket": a.get("dna", "VOLATILE"),
                                "sector": a.get("sector", "Other")}

    # ---- own regime panel (for regime-conditional stats)
    try:
        own_regime = pl.read_parquet(ROOT/"data"/"processed"/"asset_own_regime_panel.parquet").to_pandas()
        own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    except Exception:
        own_regime = pd.DataFrame()
    print(f"  own_regime panel: {len(own_regime)} rows")

    # ---- mine all cells
    print("\n[1/4] Mining cells per (asset, cadence)...")
    all_rows = []
    for asset in U50:
        for cadence in CADENCES:
            rows = mine_per_asset_cadence(cl, asset, cadence, raw_cands)
            for r in rows:
                meta = asset_meta.get(asset, {})
                r["bucket"] = meta.get("bucket", "VOLATILE")
                r["sector"] = meta.get("sector", "Other")
                all_rows.append(r)
        print(f"  {asset}: {sum(1 for r in all_rows if r['asset']==asset)} cells across {len(CADENCES)} cadences")
    cells_df = pd.DataFrame(all_rows)
    cells_df.to_parquet(OUT_DIR / "master_smart_cells_per_asset_cadence.parquet", index=False)
    print(f"\n  saved {len(cells_df)} cells to master_smart_cells_per_asset_cadence.parquet")

    # ---- regime-conditional active% (1d only)
    print("\n[2/4] Computing regime-conditional active% (1d cells)...")
    cells_1d = cells_df[cells_df["cadence"] == "1d"].copy()
    if len(own_regime):
        regime_stats = []
        # cache chimera 1d per asset
        chim_1d_cache = {}
        for asset in cells_1d["asset"].unique():
            try:
                df = cl.load(asset, "1d")
                if df is None:
                    continue
                if hasattr(df, "to_pandas"):
                    df = df.to_pandas()
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
                chim_1d_cache[asset] = df
            except Exception:
                continue
        for _, c in cells_1d.iterrows():
            df = chim_1d_cache.get(c["asset"])
            if df is None:
                regime_stats.append({"bull": 0, "chop": 0, "bear": 0, "crash": 0})
                continue
            closes = df["close"].values.astype(float)
            dates = df["date"].values
            own = own_regime[own_regime["asset"] == c["asset"]]
            rstats = regime_conditional_active(closes, dates, int(c["fast"]), int(c["slow"]),
                                                 c["ma_type"], own, TRAIN_VAL_END)
            regime_stats.append(rstats)
        cells_1d["regime_pct_active_bull"] = [r["bull"] for r in regime_stats]
        cells_1d["regime_pct_active_chop"] = [r["chop"] for r in regime_stats]
        cells_1d["regime_pct_active_bear"] = [r["bear"] for r in regime_stats]
        cells_1d["regime_pct_active_crash"] = [r["crash"] for r in regime_stats]

    # ---- top cell per (asset, cadence) by sharpe_train (qualifying min_fires)
    print("\n[3/4] Picking top cells + writing master CSV...")
    cells_df = cells_df.merge(
        cells_1d[["asset", "fast", "slow", "ma_type",
                  "regime_pct_active_bull", "regime_pct_active_chop",
                  "regime_pct_active_bear", "regime_pct_active_crash"]],
        on=["asset", "fast", "slow", "ma_type"], how="left",
    )
    qual = cells_df[cells_df["n_train"] >= MIN_FIRES_TRAIN].copy()
    top = (qual.sort_values("sharpe_train", ascending=False)
              .drop_duplicates(subset=["asset", "cadence"], keep="first")
              .reset_index(drop=True))
    top_csv_path = OUT_DIR / "MASTER_TOP_PER_ASSET_CADENCE_SMART_2026_05_20.csv"
    top.to_csv(top_csv_path, index=False)
    print(f"  top cell per (asset, cadence): {len(top)} rows -> {top_csv_path.name}")

    # ---- per-asset story (best cadence + best cell + summary)
    asset_story = []
    for asset in top["asset"].unique():
        asub = top[top["asset"] == asset].sort_values("sharpe_train", ascending=False)
        if not len(asub):
            continue
        best = asub.iloc[0]
        alt = asub.iloc[1] if len(asub) > 1 else None
        # Identify best regime
        if "regime_pct_active_bull" in best.index and pd.notna(best.get("regime_pct_active_bull")):
            r_map = {
                "bull": best.get("regime_pct_active_bull", 0),
                "chop": best.get("regime_pct_active_chop", 0),
                "bear": best.get("regime_pct_active_bear", 0),
                "crash": best.get("regime_pct_active_crash", 0),
            }
            best_regime = max(r_map, key=r_map.get)
        else:
            best_regime = "unknown"
        # Composite OOS confirmation
        oos_conf = "OOS confirmed" if best["sharpe_oos"] > 0.05 and best["n_oos"] >= MIN_FIRES_OOS_REPORT else ("OOS partial" if best["n_oos"] >= 1 else "no OOS fires")
        asset_story.append({
            "asset": asset,
            "bucket": best["bucket"],
            "sector": best["sector"],
            "best_cadence": best["cadence"],
            "best_cell": f"{best['ma_type']}({int(best['fast'])},{int(best['slow'])})",
            "n_train": int(best["n_train"]),
            "sharpe_train": float(best["sharpe_train"]),
            "hit_train": float(best["hit_train"]),
            "mean_pnl_train": float(best["mean_pnl_train"]),
            "n_oos": int(best["n_oos"]),
            "sharpe_oos": float(best["sharpe_oos"]),
            "hit_oos": float(best["hit_oos"]),
            "mean_pnl_oos": float(best["mean_pnl_oos"]),
            "oos_status": oos_conf,
            "best_regime": best_regime,
            "alt_cadence": alt["cadence"] if alt is not None else "",
            "alt_cell": f"{alt['ma_type']}({int(alt['fast'])},{int(alt['slow'])})" if alt is not None else "",
            "alt_sharpe_train": float(alt["sharpe_train"]) if alt is not None else 0.0,
        })
    story_df = pd.DataFrame(asset_story).sort_values("sharpe_train", ascending=False)
    story_csv_path = OUT_DIR / "MASTER_PER_ASSET_STORY_2026_05_20.csv"
    story_df.to_csv(story_csv_path, index=False)
    print(f"  per-asset story: {len(story_df)} rows -> {story_csv_path.name}")

    # ---- markdown story
    lines = ["# Master Per-Asset Story (smart-grid + MA-active + OOS confirmed)\n",
             f"\n**Date**: 2026-05-20  ",
             f"\n**TRAIN+VAL end**: {TRAIN_VAL_END}  ",
             f"\n**OOS**: {OOS_START} -> {OOS_END}  ",
             f"\n**Substrate**: smart grid (Fibonacci + golden + log-spaced + decorrelation, 65 raw candidates)  \n",
             "\n| Asset | Bucket | Best TF | Best Cell | n_train | Sharpe_train | Hit | Mean PnL | n_OOS | Sharpe_OOS | OOS status | Best Regime | Alt TF | Alt Cell |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for _, r in story_df.iterrows():
        lines.append(f"| {r['asset']} | {r['bucket']} | {r['best_cadence']} | "
                     f"{r['best_cell']} | {r['n_train']} | {r['sharpe_train']:+.3f} | "
                     f"{r['hit_train']*100:.1f}% | {r['mean_pnl_train']*100:+.2f}% | "
                     f"{r['n_oos']} | {r['sharpe_oos']:+.3f} | {r['oos_status']} | "
                     f"{r['best_regime']} | {r['alt_cadence']} | {r['alt_cell']} |")
    (OUT_DIR / "MASTER_PER_ASSET_STORY_2026_05_20.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote MASTER_PER_ASSET_STORY_2026_05_20.md")

    # ---- per-asset smart-grid profile (for V8 sim)
    # Filter to qualifying cells (>= MIN_FIRES_TRAIN); keep top-3 per (asset, cadence)
    qual = cells_df[cells_df["n_train"] >= MIN_FIRES_TRAIN].copy()
    qual_top = (qual.sort_values("sharpe_train", ascending=False)
                  .groupby(["asset", "cadence"]).head(3).reset_index(drop=True))
    qual_top.to_parquet(OUT_DIR / "per_asset_smart_grid_profile.parquet", index=False)
    print(f"\n  per-asset smart-grid profile (top-3 per asset×cadence, n_train>={MIN_FIRES_TRAIN}): "
          f"{len(qual_top)} cells -> per_asset_smart_grid_profile.parquet")

    # ---- summary
    print("\n[4/4] Done. Summary:")
    print(f"  Total cells mined: {len(cells_df)}")
    print(f"  Qualifying cells (n_train>={MIN_FIRES_TRAIN}): {len(qual)}")
    print(f"  Top cells (top-3 per asset×cadence): {len(qual_top)}")
    print(f"  Per-asset stories: {len(story_df)}")
    print(f"\n  outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
