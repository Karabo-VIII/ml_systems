"""Chimera mining engine -- P1 (feature catalog/health) + P2 (per-asset structure corpus).

DESCRIPTIVE / unsupervised. Streams ONE chimera file at a time (memory-safe: the full 15m x u100 panel
never lives in memory) and emits compact per-(asset,cadence) aggregates + a per-cadence feature catalog.

Outputs:
  runs/mining/feature_catalog_<cadence>.csv  -- per column: coverage across assets, missing-frac, mean|std, family
  runs/mining/corpus_<cadence>.parquet       -- per asset: return/price STRUCTURE (Hurst, VR, AC, vol, DD, trend,
                                                regime shares) + per-family feature intensity/coverage

Run:  python src/mining/chimera_mine.py --cadences 1d,4h,1h,30m,15m --universe u100
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

BARS_PER_YEAR = {"1d": 365.0, "4h": 365.0 * 6, "1h": 365.0 * 24, "30m": 365.0 * 48, "15m": 365.0 * 96}
BPD = {"1d": 1, "4h": 6, "1h": 24, "30m": 48, "15m": 96}

# feature-family prefixes (the structural partition of the 243-col schema)
FAMILIES = ["norm", "xd", "hbr", "s3", "bs", "liq", "wh", "soc", "xex", "dv", "stbl", "etf",
            "fund", "premium", "rv", "te", "mv", "lob", "bd", "xrel"]
# columns that are NOT predictor-features (price/meta/targets) -- excluded from family intensity
NON_FEATURE = {"timestamp", "bar_id", "open", "high", "low", "close", "volume", "volume_usd",
               "buy_vol", "sell_vol", "tick_count", "tick_seq", "returns_clean", "date",
               "is_u10", "is_u50", "is_u100", "asset_dna", "regime_label", "hurst_regime", "fp_fund_panel"}


def _family_of(col: str) -> str:
    pre = col.split("_")[0]
    return pre if pre in FAMILIES else ("target" if col.startswith("target_") else "other")


def _hurst(ret: np.ndarray) -> float:
    """Aggregated-variance Hurst: slope of log(std of k-aggregated returns) vs log(k). H~0.5 random,
    >0.5 persistent/trending, <0.5 mean-reverting. Robust + cheap."""
    ret = ret[np.isfinite(ret)]
    n = len(ret)
    if n < 200:
        return float("nan")
    ks = [k for k in (2, 4, 8, 16, 32, 64) if k < n // 4]
    if len(ks) < 3:
        return float("nan")
    xs, ys = [], []
    for k in ks:
        m = (n // k) * k
        agg = ret[:m].reshape(-1, k).sum(axis=1)
        s = np.std(agg)
        if s > 0:
            xs.append(np.log(k)); ys.append(np.log(s))
    if len(xs) < 3:
        return float("nan")
    # std of k-sum ~ k^H  =>  slope of log-std vs log-k = H
    return float(np.polyfit(xs, ys, 1)[0])


def _variance_ratio(ret: np.ndarray, q: int = 5) -> float:
    """VR(q) = Var(q-sum)/(q*Var(1)). >1 trending, <1 mean-reverting, ~1 random walk."""
    ret = ret[np.isfinite(ret)]
    if len(ret) < q * 10:
        return float("nan")
    v1 = np.var(ret)
    if v1 <= 0:
        return float("nan")
    m = (len(ret) // q) * q
    vq = np.var(ret[:m].reshape(-1, q).sum(axis=1))
    return float(vq / (q * v1))


def _ac1(ret: np.ndarray) -> float:
    ret = ret[np.isfinite(ret)]
    if len(ret) < 50:
        return float("nan")
    a = ret[:-1]; b = ret[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _max_dd(close: np.ndarray) -> float:
    c = close[np.isfinite(close)]
    if len(c) < 2:
        return float("nan")
    peak = np.maximum.accumulate(c)
    dd = (c - peak) / peak
    return float(dd.min() * 100.0)


def mine_file(path: str, cadence: str) -> tuple[dict, list]:
    df = pl.read_parquet(path)
    name = Path(path).stem
    sym = name.split("_v51_")[0].upper()
    close = df["close"].to_numpy().astype(float) if "close" in df.columns else np.array([])
    n = len(close)
    ret = np.zeros(n); ret[1:] = np.diff(np.log(np.clip(close, 1e-12, None))) if n > 1 else 0.0
    ret_v = ret[1:] if n > 1 else np.array([])
    bpy = BARS_PER_YEAR[cadence]

    # --- price/return STRUCTURE ---
    row = {"sym": sym, "cadence": cadence, "n_bars": n,
           "ret_mean": float(np.mean(ret_v)) if len(ret_v) else float("nan"),
           "ret_std": float(np.std(ret_v)) if len(ret_v) else float("nan"),
           "ret_skew": float(_skew(ret_v)), "ret_kurt": float(_kurt(ret_v)),
           "ann_vol_pct": float(np.std(ret_v) * np.sqrt(bpy) * 100) if len(ret_v) else float("nan"),
           "ann_ret_pct": float(np.mean(ret_v) * bpy * 100) if len(ret_v) else float("nan"),
           "max_dd_pct": _max_dd(close), "ac1": _ac1(ret_v),
           "variance_ratio_5": _variance_ratio(ret_v, 5), "hurst_aggvar": _hurst(ret_v)}
    # trend fraction: time close > SMA(200 native bars)
    if n >= 200:
        sma = pl.Series(close).rolling_mean(200).to_numpy()
        valid = np.isfinite(sma)
        row["trend_frac_above_sma200"] = float(np.mean(close[valid] > sma[valid])) if valid.any() else float("nan")
    else:
        row["trend_frac_above_sma200"] = float("nan")
    # existing regime_label shares + mean hurst_regime
    if "regime_label" in df.columns:
        rl = df["regime_label"].to_numpy()
        rl = rl[np.isfinite(rl.astype(float))] if rl.dtype != object else rl
        for k in (0, 1, 2):
            row[f"regime{k}_share"] = float(np.mean(rl == k)) if len(rl) else float("nan")
    if "hurst_regime" in df.columns:
        hr = df["hurst_regime"].to_numpy().astype(float)
        row["hurst_regime_mean"] = float(np.nanmean(hr)) if len(hr) else float("nan")

    # --- per-FAMILY feature intensity + coverage ---
    catalog = []
    fam_absmean = {f: [] for f in FAMILIES}
    fam_cover = {f: [] for f in FAMILIES}
    for col in df.columns:
        if col in NON_FEATURE or col.startswith("target_"):
            # still catalog targets/meta health but don't fold into family intensity
            fam = _family_of(col)
        else:
            fam = _family_of(col)
        s = df[col]
        try:
            x = s.to_numpy().astype(float)
        except Exception:
            continue
        nn = int(np.isfinite(x).sum())
        miss = 1.0 - nn / max(n, 1)
        mu = float(np.nanmean(x)) if nn else float("nan")
        sd = float(np.nanstd(x)) if nn else float("nan")
        catalog.append({"col": col, "family": fam, "sym": sym, "n_nonnull": nn,
                        "missing_frac": round(miss, 4), "mean": mu, "std": sd,
                        "is_constant": bool(nn > 0 and sd == 0)})
        if fam in fam_absmean and col not in NON_FEATURE and not col.startswith("target_"):
            if nn:
                fam_absmean[fam].append(float(np.nanmean(np.abs(x))))
                fam_cover[fam].append(nn / max(n, 1))
    for f in FAMILIES:
        row[f"fam_{f}_absmean"] = float(np.mean(fam_absmean[f])) if fam_absmean[f] else float("nan")
        row[f"fam_{f}_cover"] = float(np.mean(fam_cover[f])) if fam_cover[f] else 0.0
    return row, catalog


def _skew(x):
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) == 0:
        return float("nan")
    z = (x - np.mean(x)) / np.std(x)
    return float(np.mean(z ** 3))


def _kurt(x):
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.std(x) == 0:
        return float("nan")
    z = (x - np.mean(x)) / np.std(x)
    return float(np.mean(z ** 4) - 3.0)


def mine_cadence(cadence: str, universe: str = "u100") -> dict:
    files = sorted(glob.glob(str(ROOT / "data" / "processed" / "chimera" / cadence /
                                 f"*_v51_chimera_{cadence}_*.parquet")))
    if not files:
        return {"cadence": cadence, "error": "no files"}
    rows, catalog_all = [], []
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            row, cat = mine_file(f, cadence)
            rows.append(row); catalog_all.extend(cat)
        except Exception as e:
            print(f"  [skip] {Path(f).name}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  {cadence}: {i+1}/{len(files)} files ({time.time()-t0:.0f}s)", flush=True)
    corpus = pl.DataFrame(rows)
    corpus.write_parquet(OUT / f"corpus_{cadence}.parquet")
    # aggregate catalog across assets -> per (col,family): coverage, mean missing, mean|std
    cat = pl.DataFrame(catalog_all)
    cat_agg = (cat.group_by(["col", "family"]).agg([
        pl.len().alias("n_assets_present"),
        pl.col("missing_frac").mean().round(4).alias("mean_missing_frac"),
        pl.col("mean").mean().alias("mean_of_means"),
        pl.col("std").mean().alias("mean_of_stds"),
        pl.col("is_constant").sum().alias("n_constant_assets"),
    ]).sort(["family", "col"]))
    cat_agg.write_csv(OUT / f"feature_catalog_{cadence}.csv")
    summary = {"cadence": cadence, "n_assets": len(rows), "n_cols": cat["col"].n_unique(),
               "elapsed_s": round(time.time() - t0, 1),
               "corpus": str(OUT / f"corpus_{cadence}.parquet"),
               "catalog": str(OUT / f"feature_catalog_{cadence}.csv")}
    print(f"=== {cadence}: {summary['n_assets']} assets, {summary['n_cols']} cols, {summary['elapsed_s']}s ===", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", default="1d,4h,1h,30m,15m")
    ap.add_argument("--universe", default="u100")
    a = ap.parse_args(argv)
    manifest_p = OUT / "mine_manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    for cad in [c.strip() for c in a.cadences.split(",") if c.strip()]:
        print(f"\n##### MINING CADENCE {cad} #####", flush=True)
        manifest[cad] = mine_cadence(cad, a.universe)
        manifest_p.write_text(json.dumps(manifest, indent=2, default=str))  # incremental
    print("\n[manifest]", manifest_p)


if __name__ == "__main__":
    raise SystemExit(main())
