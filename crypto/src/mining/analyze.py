"""Chimera mining analyses -- P3 (regime decomposition), P4 (asset clustering), P5 (trend + feature structure).

Consumes the P2 corpus (runs/mining/corpus_<cad>.parquet) + streams chimera files for the per-bar needs
(regime fit, returns matrix, feature PCA). Memory-safe: caps pooled rows; resamples returns to daily for
the correlation matrix on fine cadences. DESCRIPTIVE only -- no predictive/alpha claim (honesty rail).

Outputs per cadence:
  runs/mining/regimes_<cad>.json       -- discovered (GMM) + rule (regime_label) regimes, shares, transitions, profiles
  runs/mining/asset_clusters_<cad>.json -- return-corr hierarchical clusters, BTC-beta, BTC lead-lag
  runs/mining/structure_<cad>.json     -- trend-vs-meanrevert map, feature redundancy + PCA effective-dim + top loadings

Run:  python src/mining/analyze.py --cadences 1d,4h,1h,30m,15m
No emoji (cp1252).
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
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "mining"
BPD = {"1d": 1, "4h": 6, "1h": 24, "30m": 48, "15m": 96}
DENSE_FEATS_PREFIX = ("norm_", "xd_")  # universe-wide dense features for PCA / redundancy


def _files(cadence):
    return sorted(glob.glob(str(ROOT / "data" / "processed" / "chimera" / cadence /
                                f"*_v51_chimera_{cadence}_*.parquet")))


def _sym(path):
    return Path(path).stem.split("_v51_")[0].upper()


# ------------------------------------------------------------------ P3 regime
def regime_decomposition(cadence: str, max_rows_per_asset: int = 3000) -> dict:
    files = _files(cadence)
    pooled = []          # [ret, logvol, trend] per-bar, standardized later
    btc_seq = None       # BTC regime_label sequence for transition matrix
    rl_trans = np.zeros((3, 3))   # existing regime_label transition counts (pooled)
    rl_fwd = {0: [], 1: [], 2: []}  # regime_label[t] -> ret[t+1] (descriptive)
    for f in files:
        try:
            df = pl.read_parquet(f, columns=["close", "regime_label"])
        except Exception:
            try:
                df = pl.read_parquet(f, columns=["close"])
            except Exception:
                continue
        close = df["close"].to_numpy().astype(float)
        n = len(close)
        if n < 250:
            continue
        ret = np.zeros(n); ret[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
        vol = pd.Series(ret).rolling(20).std().to_numpy()
        sma = pd.Series(close).rolling(50).mean().to_numpy()
        trend = (close - sma) / np.where(sma > 0, sma, np.nan)
        feat = np.column_stack([ret, vol, trend])
        ok = np.isfinite(feat).all(axis=1)
        feat = feat[ok]
        if len(feat) > max_rows_per_asset:
            idx = np.linspace(0, len(feat) - 1, max_rows_per_asset).astype(int)
            feat = feat[idx]
        pooled.append(feat)
        if "regime_label" in df.columns:
            rl = df["regime_label"].to_numpy()
            try:
                rl = rl.astype(int)
                for a, b in zip(rl[:-1], rl[1:]):
                    if 0 <= a < 3 and 0 <= b < 3:
                        rl_trans[a, b] += 1
                for t in range(len(rl) - 1):
                    if 0 <= rl[t] < 3 and np.isfinite(ret[t + 1]):
                        rl_fwd[int(rl[t])].append(ret[t + 1])
                if _sym(f) == "BTCUSDT":
                    btc_seq = rl
            except Exception:
                pass
    if not pooled:
        return {"cadence": cadence, "error": "no pooled regime data"}
    X = np.vstack(pooled)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    # discovered regimes: GMM, pick k by BIC over 2..5
    best = None
    for k in (2, 3, 4, 5):
        try:
            gm = GaussianMixture(k, covariance_type="full", random_state=0, n_init=2, max_iter=200).fit(Xs)
            bic = gm.bic(Xs)
            if best is None or bic < best[0]:
                best = (bic, k, gm)
        except Exception:
            continue
    out = {"cadence": cadence, "n_pooled_bars": int(len(X)),
           "discovered_features": ["ret", "rolling_vol20", "trend_vs_sma50"]}
    if best:
        bic, k, gm = best
        lab = gm.predict(Xs)
        profiles = []
        for r in range(k):
            m = lab == r
            profiles.append({"regime": int(r), "share": round(float(m.mean()), 3),
                             "ret_mean_bar": float(np.mean(X[m, 0])), "vol_mean": float(np.mean(X[m, 1])),
                             "trend_mean": float(np.mean(X[m, 2]))})
        # sort regimes by trend_mean for readability (bear -> bull)
        profiles.sort(key=lambda p: p["trend_mean"])
        out["discovered_gmm"] = {"k_by_bic": int(k), "bic": round(float(bic), 1), "profiles": profiles}
    # existing regime_label
    tot = rl_trans.sum(1, keepdims=True); tot[tot == 0] = 1
    out["regime_label_transition_matrix"] = (rl_trans / tot).round(3).tolist()
    out["regime_label_persistence_diag"] = [round(float((rl_trans / tot)[i, i]), 3) for i in range(3)]
    out["regime_label_fwd_ret_bar"] = {str(k): (round(float(np.mean(v)), 6) if v else None) for k, v in rl_fwd.items()}
    out["note"] = "regime_label[t]->ret[t+1] is DESCRIPTIVE co-occurrence, not a validated predictor."
    return out


# ------------------------------------------------------------------ P4 clustering
def _returns_matrix(cadence: str, resample_daily_if_big: bool = True) -> pd.DataFrame:
    files = _files(cadence)
    big = BPD[cadence] >= 24
    series = {}
    for f in files:
        try:
            df = pl.read_parquet(f, columns=["date", "close"])
        except Exception:
            continue
        pdf = df.to_pandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.dropna().drop_duplicates("date").set_index("date")["close"]
        if big and resample_daily_if_big:
            pdf = pdf.resample("1D").last().dropna()
        series[_sym(f)] = np.log(pdf).diff()
    mat = pd.DataFrame(series).dropna(how="all")
    return mat


def asset_clustering(cadence: str) -> dict:
    mat = _returns_matrix(cadence)
    # keep assets with enough overlap
    mat = mat.dropna(axis=1, thresh=int(0.5 * len(mat)))
    corr = mat.corr(min_periods=60)
    corr = corr.dropna(how="all").dropna(axis=1, how="all")
    assets = list(corr.index)
    C = corr.fillna(0).to_numpy()
    np.fill_diagonal(C, 1.0)
    out = {"cadence": cadence, "n_assets": len(assets),
           "median_pairwise_corr": round(float(np.median(C[np.triu_indices_from(C, 1)])), 3)}
    # hierarchical clustering on (1-corr) distance
    try:
        dist = 1.0 - C
        np.fill_diagonal(dist, 0.0)
        from scipy.spatial.distance import squareform
        Z = linkage(squareform(dist, checks=False), method="average")
        for kk in (4, 6, 8):
            lab = fcluster(Z, kk, criterion="maxclust")
            clusters = {}
            for a, l in zip(assets, lab):
                clusters.setdefault(int(l), []).append(a)
            out[f"clusters_k{kk}"] = {str(k): v for k, v in sorted(clusters.items())}
    except Exception as e:
        out["cluster_error"] = f"{type(e).__name__}: {e}"
    # BTC-beta + lead-lag
    if "BTCUSDT" in mat.columns:
        btc = mat["BTCUSDT"]
        betas, leadlag = {}, {}
        for a in mat.columns:
            if a == "BTCUSDT":
                continue
            j = pd.concat([btc, mat[a]], axis=1).dropna()
            if len(j) < 100:
                continue
            b = np.polyfit(j.iloc[:, 0], j.iloc[:, 1], 1)[0]
            betas[a] = round(float(b), 3)
            # lead-lag: corr(btc[t], alt[t+lag]); positive best-lag => BTC leads
            best_lag, best_c = 0, 0.0
            for lag in range(-3, 4):
                cc = j.iloc[:, 0].corr(j.iloc[:, 1].shift(-lag))
                if np.isfinite(cc) and abs(cc) > abs(best_c):
                    best_c, best_lag = cc, lag
            leadlag[a] = best_lag
        out["btc_beta_median"] = round(float(np.median(list(betas.values()))), 3) if betas else None
        out["btc_beta_sample"] = dict(sorted(betas.items(), key=lambda x: -x[1])[:10])
        ll = [v for v in leadlag.values()]
        out["btc_leads_frac"] = round(sum(1 for v in ll if v > 0) / max(len(ll), 1), 3) if ll else None
        out["alt_leads_frac"] = round(sum(1 for v in ll if v < 0) / max(len(ll), 1), 3) if ll else None
        out["contemporaneous_frac"] = round(sum(1 for v in ll if v == 0) / max(len(ll), 1), 3) if ll else None
        out["leadlag_note"] = "best_lag>0 means BTC[t] aligns with alt[t+lag] -> BTC leads."
    return out


# ------------------------------------------------------------------ P5 structure
def trend_and_feature_structure(cadence: str, sample_assets: int = 25, max_rows: int = 120000) -> dict:
    cpath = OUT / f"corpus_{cadence}.parquet"
    out = {"cadence": cadence}
    # trend-vs-meanrevert map from corpus
    if cpath.exists():
        c = pl.read_parquet(cpath)
        h = c["hurst_aggvar"].to_numpy(); vr = c["variance_ratio_5"].to_numpy()
        h = h[np.isfinite(h)]
        trd = int(np.sum(h > 0.55)); mr = int(np.sum(h < 0.45)); rw = int(np.sum((h >= 0.45) & (h <= 0.55)))
        out["trend_character"] = {"n_assets": int(len(h)), "trending_H>0.55": trd,
                                  "meanrev_H<0.45": mr, "randomwalk_0.45-0.55": rw,
                                  "median_hurst": round(float(np.median(h)), 3),
                                  "median_VR5": round(float(np.nanmedian(vr)), 3)}
    # feature redundancy + PCA on dense features (norm_/xd_), pooled across a sample of assets
    files = _files(cadence)
    if len(files) > sample_assets:
        idx = np.linspace(0, len(files) - 1, sample_assets).astype(int)
        files = [files[i] for i in idx]
    rows, feat_cols = [], None
    for f in files:
        try:
            df = pl.read_parquet(f)
        except Exception:
            continue
        cols = [c for c in df.columns if c.startswith(DENSE_FEATS_PREFIX)]
        if feat_cols is None:
            feat_cols = cols
        cols = [c for c in feat_cols if c in df.columns]
        sub = df.select(cols).to_numpy().astype(float)
        rows.append(sub)
        if sum(len(r) for r in rows) > max_rows:
            break
    if not rows or not feat_cols:
        out["structure_error"] = "no dense feature rows"
        return out
    X = np.vstack(rows)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    if len(X) > max_rows:
        X = X[np.linspace(0, len(X) - 1, max_rows).astype(int)]
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    # PCA effective dimensionality
    p = PCA().fit(Xs)
    evr = p.explained_variance_ratio_
    cum = np.cumsum(evr)
    out["feature_pca"] = {"n_dense_features": len(feat_cols), "n_rows": int(len(X)),
                          "pcs_for_70pct": int(np.searchsorted(cum, 0.70) + 1),
                          "pcs_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
                          "pc1_var": round(float(evr[0]), 3), "pc2_var": round(float(evr[1]), 3),
                          "top_pc1_loadings": _top_loadings(p.components_[0], feat_cols),
                          "top_pc2_loadings": _top_loadings(p.components_[1], feat_cols)}
    # redundancy: count |corr|>0.8 feature pairs
    C = np.corrcoef(Xs.T)
    iu = np.triu_indices_from(C, 1)
    out["feature_redundancy"] = {"n_pairs": int(len(iu[0])),
                                 "n_pairs_abscorr_gt_0.8": int(np.sum(np.abs(C[iu]) > 0.8)),
                                 "n_pairs_abscorr_gt_0.6": int(np.sum(np.abs(C[iu]) > 0.6))}
    return out


def _top_loadings(comp, cols, k=6):
    order = np.argsort(-np.abs(comp))[:k]
    return [{"feat": cols[i], "loading": round(float(comp[i]), 3)} for i in order]


# ------------------------------------------------------------------ driver
def run_cadence(cadence: str) -> dict:
    res = {}
    for name, fn in [("regimes", regime_decomposition), ("asset_clusters", asset_clustering),
                     ("structure", trend_and_feature_structure)]:
        t0 = time.time()
        try:
            r = fn(cadence)
        except Exception as e:
            r = {"cadence": cadence, "error": f"{type(e).__name__}: {e}"}
        r["_elapsed_s"] = round(time.time() - t0, 1)
        (OUT / f"{name}_{cadence}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        res[name] = r
        print(f"  [{cadence}] {name}: {r.get('_elapsed_s')}s {'ERROR '+r['error'] if 'error' in r else 'ok'}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", default="1d,4h,1h,30m,15m")
    a = ap.parse_args(argv)
    for cad in [c.strip() for c in a.cadences.split(",") if c.strip()]:
        if not (OUT / f"corpus_{cad}.parquet").exists():
            print(f"[skip {cad}] corpus not built yet", flush=True); continue
        print(f"\n##### ANALYZE {cad} #####", flush=True)
        run_cadence(cad)


if __name__ == "__main__":
    raise SystemExit(main())
