"""Chimera DEEP mining -- P7-P9 (beyond the P3-P5 structure pass).

P7  vol_clustering        : is volatility persistent (the one robustly predictable structure)? AC1 of squared returns.
P7b regime_feature_signatures : per existing regime_label (0/1/2), which dense features are characteristically hi/lo.
P8  mr_trend_candidates   : rank assets by Hurst/AC1 -> most mean-reverting vs most trending (actionable lists).
P8b cross_sectional_dispersion : per-bar cross-asset return dispersion + count moving >5%/>10% (the movers premise).
P9  feature_move_screen   : DESCRIPTIVE association of each dense feature (as-of bar t) with next-bar |return| and
                            next-bar sign, pooled across assets. Broadened oracle-decomposer DNA. HYPOTHESIS-GEN ONLY.

Memory-safe (stream, accumulate aggregates). DESCRIPTIVE -- no gate-validated claim. No emoji.
Run:  python src/mining/deep_mine.py --cadences 1d,4h,1h,30m,15m
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "mining"
DENSE_PREFIX = ("norm_", "xd_")


def _files(cad):
    return sorted(glob.glob(str(ROOT / "data" / "processed" / "chimera" / cad /
                                f"*_v51_chimera_{cad}_*.parquet")))


def _sym(p):
    return Path(p).stem.split("_v51_")[0].upper()


def _ac1(x):
    x = x[np.isfinite(x)]
    if len(x) < 50 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


# -------------------------------------------------------- P7 vol clustering
def vol_clustering(cad: str) -> dict:
    ac_sq, ac_abs = [], []
    for f in _files(cad):
        try:
            close = pl.read_parquet(f, columns=["close"])["close"].to_numpy().astype(float)
        except Exception:
            continue
        if len(close) < 250:
            continue
        ret = np.diff(np.log(np.clip(close, 1e-12, None)))
        ac_sq.append(_ac1(ret ** 2))
        ac_abs.append(_ac1(np.abs(ret)))
    ac_sq = np.array([a for a in ac_sq if np.isfinite(a)])
    ac_abs = np.array([a for a in ac_abs if np.isfinite(a)])
    return {"cadence": cad, "n_assets": int(len(ac_sq)),
            "median_AC1_squared_ret": round(float(np.median(ac_sq)), 4) if len(ac_sq) else None,
            "median_AC1_abs_ret": round(float(np.median(ac_abs)), 4) if len(ac_abs) else None,
            "frac_assets_volpersist_pos": round(float(np.mean(ac_abs > 0)), 3) if len(ac_abs) else None,
            "frac_assets_volpersist_gt_0.1": round(float(np.mean(ac_abs > 0.1)), 3) if len(ac_abs) else None,
            "note": "AC1 of |ret| / ret^2 > 0 => volatility clusters (today's vol predicts tomorrow's). "
                    "This is the structurally PREDICTABLE part of crypto (unlike direction)."}


# -------------------------------------------------------- P7b regime feature signatures
def regime_feature_signatures(cad: str, sample_assets: int = 40) -> dict:
    files = _files(cad)
    if len(files) > sample_assets:
        idx = np.linspace(0, len(files) - 1, sample_assets).astype(int)
        files = [files[i] for i in idx]
    feat_cols = None
    sums = {0: None, 1: None, 2: None}
    cnts = {0: 0, 1: 0, 2: 0}
    gsum = None; gcnt = 0; gsq = None
    for f in files:
        try:
            df = pl.read_parquet(f)
        except Exception:
            continue
        if "regime_label" not in df.columns:
            continue
        if feat_cols is None:
            feat_cols = [c for c in df.columns if c.startswith(DENSE_PREFIX)]
        cols = [c for c in feat_cols if c in df.columns]
        X = df.select(cols).to_numpy().astype(float)
        rl = df["regime_label"].to_numpy()
        ok = np.isfinite(X).all(axis=1)
        X = X[ok]; rl = rl[ok]
        if gsum is None:
            gsum = np.zeros(X.shape[1]); gsq = np.zeros(X.shape[1])
        gsum += X.sum(0); gsq += (X ** 2).sum(0); gcnt += len(X)
        for r in (0, 1, 2):
            m = rl == r
            if m.any():
                s = X[m].sum(0)
                sums[r] = s if sums[r] is None else sums[r] + s
                cnts[r] += int(m.sum())
    if feat_cols is None or gcnt == 0:
        return {"cadence": cad, "error": "no regime/feature data"}
    gmean = gsum / gcnt
    gstd = np.sqrt(np.maximum(gsq / gcnt - gmean ** 2, 1e-12))
    out = {"cadence": cad, "n_features": len(feat_cols), "regime_counts": cnts}
    for r in (0, 1, 2):
        if sums[r] is None or cnts[r] == 0:
            continue
        rmean = sums[r] / cnts[r]
        z = (rmean - gmean) / gstd            # how far each feature deviates in this regime (z vs global)
        order = np.argsort(-np.abs(z))[:8]
        out[f"regime{r}_signature"] = [{"feat": feat_cols[i], "z": round(float(z[i]), 2)} for i in order]
    out["note"] = "z = (regime-mean - global-mean)/global-std per dense feature. Which microstructure features "
    out["note"] += "characterize each existing regime_label state. Descriptive."
    return out


# -------------------------------------------------------- P8 MR / trend candidates
def mr_trend_candidates(cad: str) -> dict:
    cpath = OUT / f"corpus_{cad}.parquet"
    if not cpath.exists():
        return {"cadence": cad, "error": "no corpus"}
    c = pl.read_parquet(cpath).to_pandas()
    c = c[np.isfinite(c["hurst_aggvar"]) & np.isfinite(c["ac1"])]
    mr = c.sort_values("ac1").head(12)[["sym", "ac1", "hurst_aggvar", "variance_ratio_5", "ann_vol_pct"]]
    tr = c.sort_values("hurst_aggvar", ascending=False).head(12)[["sym", "hurst_aggvar", "ac1", "variance_ratio_5", "ann_vol_pct"]]
    return {"cadence": cad,
            "most_mean_reverting": [{"sym": r.sym, "ac1": round(r.ac1, 4), "hurst": round(r.hurst_aggvar, 3)}
                                    for r in mr.itertuples()],
            "most_trending": [{"sym": r.sym, "hurst": round(r.hurst_aggvar, 3), "ac1": round(r.ac1, 4)}
                              for r in tr.itertuples()],
            "note": "MR candidates = most-negative bar AC1 (caveat: bid-ask bounce). Trending = highest Hurst."}


# -------------------------------------------------------- P8b cross-sectional dispersion / movers
def cross_sectional_dispersion(cad: str) -> dict:
    files = _files(cad)
    series = {}
    for f in files:
        try:
            df = pl.read_parquet(f, columns=["date", "close"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["date"])
        s = df.dropna().drop_duplicates("date").set_index("date")["close"]
        if cad != "1d":
            s = s.resample("1D").last()
        series[_sym(f)] = np.log(s).diff()
    mat = pd.DataFrame(series)
    mat = mat[mat.notna().sum(axis=1) >= 10]   # days with >=10 assets
    if mat.empty:
        return {"cadence": cad, "error": "no dispersion data"}
    daily_disp = mat.std(axis=1)               # cross-asset return std per day
    movers5 = (mat.abs() > 0.05).sum(axis=1)    # count moving >5% (daily)
    movers10 = (mat.abs() > 0.10).sum(axis=1)
    up5 = (mat > 0.05).sum(axis=1)
    return {"cadence": cad, "n_days": int(len(mat)), "n_assets_avg": round(float(mat.notna().sum(axis=1).mean()), 1),
            "median_cross_sectional_disp": round(float(daily_disp.median()), 4),
            "median_movers_gt5pct_per_day": round(float(movers5.median()), 1),
            "median_upmovers_gt5pct_per_day": round(float(up5.median()), 1),
            "median_movers_gt10pct_per_day": round(float(movers10.median()), 1),
            "frac_days_ge1_upmover5": round(float((up5 >= 1).mean()), 3),
            "note": "the OPPORTUNITY premise quantified fresh: how dispersed is the cross-section daily."}


# -------------------------------------------------------- P9 feature -> forward-move descriptive screen
def feature_move_screen(cad: str, sample_assets: int = 40, max_rows_per_asset: int = 4000) -> dict:
    files = _files(cad)
    if len(files) > sample_assets:
        idx = np.linspace(0, len(files) - 1, sample_assets).astype(int)
        files = [files[i] for i in idx]
    feat_cols = None
    X_all, ymag_all, ysign_all = [], [], []
    for f in files:
        try:
            df = pl.read_parquet(f)
        except Exception:
            continue
        if feat_cols is None:
            feat_cols = [c for c in df.columns if c.startswith(DENSE_PREFIX)]
        cols = [c for c in feat_cols if c in df.columns]
        if "close" not in df.columns:
            continue
        X = df.select(cols).to_numpy().astype(float)
        close = df["close"].to_numpy().astype(float)
        fwd = np.full(len(close), np.nan)
        fwd[:-1] = np.diff(np.log(np.clip(close, 1e-12, None)))  # next-bar log return (aligned: feat[t] -> ret[t->t+1])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(fwd)
        X = X[ok]; fwd = fwd[ok]
        if len(X) > max_rows_per_asset:
            sel = np.linspace(0, len(X) - 1, max_rows_per_asset).astype(int)
            X = X[sel]; fwd = fwd[sel]
        X_all.append(X); ymag_all.append(np.abs(fwd)); ysign_all.append((fwd > 0).astype(float))
    if not X_all:
        return {"cadence": cad, "error": "no data"}
    X = np.vstack(X_all); ymag = np.concatenate(ymag_all); ysign = np.concatenate(ysign_all)
    # standardize features
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    # |corr| of each feature with next-bar |return| (vol-predict) and point-biserial-ish with sign (direction)
    mag_corr, sign_corr = [], []
    ym = (ymag - ymag.mean()) / (ymag.std() + 1e-12)
    ys = (ysign - ysign.mean()) / (ysign.std() + 1e-12)
    for j in range(Xs.shape[1]):
        mag_corr.append(float(np.mean(Xs[:, j] * ym)))
        sign_corr.append(float(np.mean(Xs[:, j] * ys)))
    mag_corr = np.array(mag_corr); sign_corr = np.array(sign_corr)
    om = np.argsort(-np.abs(mag_corr))[:10]
    os_ = np.argsort(-np.abs(sign_corr))[:10]
    return {"cadence": cad, "n_rows": int(len(X)), "n_features": len(feat_cols),
            "max_abs_corr_with_next_move_MAGNITUDE": round(float(np.max(np.abs(mag_corr))), 4),
            "max_abs_corr_with_next_move_DIRECTION": round(float(np.max(np.abs(sign_corr))), 4),
            "top_magnitude_predictors": [{"feat": feat_cols[i], "corr": round(float(mag_corr[i]), 4)} for i in om],
            "top_direction_predictors": [{"feat": feat_cols[i], "corr": round(float(sign_corr[i]), 4)} for i in os_],
            "note": "DESCRIPTIVE pooled correlation, feat[t] vs ret[t->t+1]. MAGNITUDE (|ret|) is the vol-predict "
                    "channel; DIRECTION (sign) is the hard alpha channel. Linear screen only; NOT gate-validated; "
                    "no multiple-testing correction applied -- hypothesis generation."}


def run_cadence(cad: str) -> dict:
    res = {}
    for name, fn in [("vol_clustering", vol_clustering), ("regime_signatures", regime_feature_signatures),
                     ("mr_trend_candidates", mr_trend_candidates),
                     ("dispersion", cross_sectional_dispersion), ("feature_move_screen", feature_move_screen)]:
        t0 = time.time()
        try:
            r = fn(cad)
        except Exception as e:
            r = {"cadence": cad, "error": f"{type(e).__name__}: {e}"}
        r["_elapsed_s"] = round(time.time() - t0, 1)
        (OUT / f"deep_{name}_{cad}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        res[name] = r
        print(f"  [{cad}] {name}: {r.get('_elapsed_s')}s {'ERROR '+str(r.get('error')) if 'error' in r else 'ok'}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", default="1d,4h,1h,30m,15m")
    a = ap.parse_args(argv)
    for cad in [c.strip() for c in a.cadences.split(",") if c.strip()]:
        print(f"\n##### DEEP MINE {cad} #####", flush=True)
        run_cadence(cad)


if __name__ == "__main__":
    raise SystemExit(main())
