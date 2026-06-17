"""Chimera mining P11 -- the predictability CEILING probe (the key open question).

Q1 nonlinear_predictability: is next-bar DIRECTION predictable nonlinearly, or is the linear ~0.04 a real
   ceiling? Pool dense features across assets, TIME-split (train=first 70%, test=last 30% -> held-out, no
   look-ahead), fit HistGradientBoosting on (a) direction sign [AUC] and (b) |return| magnitude [R2], compare
   to a linear/logistic baseline. AUC ~0.50 => direction fundamentally unpredictable; >0.53 => a nonlinear lead.
Q2 native_leadlag: does BTC lead alts at NATIVE intraday resolution (not daily-resampled)? cross-corr at lags.

DESCRIPTIVE predictability ceiling -- NOT a strategy, NO costs, NO gate, pooled (not per-asset). Hypothesis-gen.
Memory-safe (sampled rows). No emoji.
Run:  python src/mining/predictability.py --cadences 1d,4h,1h,30m,15m
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "mining"
DENSE_PREFIX = ("norm_", "xd_")


def _files(cad):
    return sorted(glob.glob(str(ROOT / "data" / "processed" / "chimera" / cad /
                                f"*_v51_chimera_{cad}_*.parquet")))


def _sym(p):
    return Path(p).stem.split("_v51_")[0].upper()


def nonlinear_predictability(cad: str, sample_assets: int = 40, max_rows_per_asset: int = 5000) -> dict:
    files = _files(cad)
    if len(files) > sample_assets:
        idx = np.linspace(0, len(files) - 1, sample_assets).astype(int)
        files = [files[i] for i in idx]
    feat_cols = None
    Xtr, Xte, ytr_s, yte_s, ytr_m, yte_m = [], [], [], [], [], []
    for f in files:
        try:
            df = pl.read_parquet(f)
        except Exception:
            continue
        if feat_cols is None:
            feat_cols = [c for c in df.columns if c.startswith(DENSE_PREFIX)]
        cols = [c for c in feat_cols if c in df.columns]
        if "close" not in df.columns or len(df) < 500:
            continue
        X = df.select(cols).to_numpy().astype(float)
        close = df["close"].to_numpy().astype(float)
        fwd = np.full(len(close), np.nan); fwd[:-1] = np.diff(np.log(np.clip(close, 1e-12, None)))
        ok = np.isfinite(fwd)
        X = X[ok]; fwd = fwd[ok]
        # enforce per-asset row cap BEFORE vstack (memory) -- even subsample preserves time order
        if len(X) > max_rows_per_asset:
            sel = np.linspace(0, len(X) - 1, max_rows_per_asset).astype(int)
            X = X[sel]; fwd = fwd[sel]
        # time split per asset (preserves no-look-ahead): first 70% train, last 30% test
        cut = int(0.7 * len(X))
        Xtr.append(X[:cut]); Xte.append(X[cut:])
        ytr_s.append((fwd[:cut] > 0).astype(int)); yte_s.append((fwd[cut:] > 0).astype(int))
        ytr_m.append(np.abs(fwd[:cut])); yte_m.append(np.abs(fwd[cut:]))
    if not Xtr:
        return {"cadence": cad, "error": "no data"}
    Xtr = np.vstack(Xtr); Xte = np.vstack(Xte)
    ytr_s = np.concatenate(ytr_s); yte_s = np.concatenate(yte_s)
    ytr_m = np.concatenate(ytr_m); yte_m = np.concatenate(yte_m)
    # subsample for speed
    rng = np.random.RandomState(0)
    def _sub(n, cap=200000):
        return rng.choice(n, cap, replace=False) if n > cap else np.arange(n)
    itr = _sub(len(Xtr)); ite = _sub(len(Xte))
    Xtr, ytr_s, ytr_m = Xtr[itr], ytr_s[itr], ytr_m[itr]
    Xte, yte_s, yte_m = Xte[ite], yte_s[ite], yte_m[ite]
    out = {"cadence": cad, "n_features": len(feat_cols), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
           "test_base_rate_up": round(float(yte_s.mean()), 3)}
    # DIRECTION: GBM AUC + logistic baseline AUC (held-out)
    try:
        gb = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05,
                                            l2_regularization=1.0, random_state=0).fit(Xtr, ytr_s)
        out["direction_AUC_gbm"] = round(float(roc_auc_score(yte_s, gb.predict_proba(Xte)[:, 1])), 4)
    except Exception as e:
        out["direction_AUC_gbm"] = f"err {type(e).__name__}"
    try:
        Xtr2 = np.nan_to_num(Xtr); Xte2 = np.nan_to_num(Xte)
        lr = LogisticRegression(max_iter=300, C=0.5).fit(Xtr2, ytr_s)
        out["direction_AUC_logistic"] = round(float(roc_auc_score(yte_s, lr.predict_proba(Xte2)[:, 1])), 4)
    except Exception as e:
        out["direction_AUC_logistic"] = f"err {type(e).__name__}"
    # MAGNITUDE: GBM R2 (held-out) -- the predictable channel for contrast
    try:
        gm = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=0).fit(Xtr, ytr_m)
        out["magnitude_R2_gbm"] = round(float(r2_score(yte_m, gm.predict(Xte))), 4)
    except Exception as e:
        out["magnitude_R2_gbm"] = f"err {type(e).__name__}"
    out["verdict"] = ("DIRECTION unpredictable (AUC~0.5)" if isinstance(out.get("direction_AUC_gbm"), float)
                      and out["direction_AUC_gbm"] < 0.53 else "DIRECTION shows >0.53 AUC -- investigate")
    out["note"] = ("Held-out time-split (train first 70%, test last 30%), pooled across assets. AUC 0.50=coin flip. "
                   "Magnitude R2>0 confirms vol predictability. NO costs/gate -- predictability ceiling only.")
    return out


_INTERVAL_MS = {"1d": 86400000, "4h": 14400000, "1h": 3600000, "30m": 1800000, "15m": 900000}


def native_leadlag(cad: str, sample_assets: int = 30, max_lag: int = 5) -> dict:
    files = _files(cad)
    iv = _INTERVAL_MS[cad]
    btc_f = [f for f in files if _sym(f) == "BTCUSDT"]
    if not btc_f:
        return {"cadence": cad, "error": "no BTC"}
    btc = pl.read_parquet(btc_f[0], columns=["timestamp", "close"]).to_pandas()
    btc["g"] = (btc["timestamp"] // iv) * iv          # floor last-trade ts to the bar grid -> common key
    btc["r"] = np.log(btc["close"]).diff()
    btc = btc[["g", "r"]].rename(columns={"r": "btc_r"}).dropna().drop_duplicates("g")
    others = [f for f in files if _sym(f) != "BTCUSDT"]
    if len(others) > sample_assets:
        idx = np.linspace(0, len(others) - 1, sample_assets).astype(int)
        others = [others[i] for i in idx]
    best_lags = []
    for f in others:
        try:
            a = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        a["g"] = (a["timestamp"] // iv) * iv
        a["alt_r"] = np.log(a["close"]).diff()
        a = a[["g", "alt_r"]].dropna().drop_duplicates("g")
        j = btc.merge(a, on="g", how="inner")
        if len(j) < 500:
            continue
        best_c, best_lag = 0.0, 0
        for lag in range(-max_lag, max_lag + 1):
            cc = j["btc_r"].corr(j["alt_r"].shift(-lag))   # btc[t] vs alt[t+lag]; lag>0 => BTC leads
            if np.isfinite(cc) and abs(cc) > abs(best_c):
                best_c, best_lag = cc, lag
        best_lags.append(best_lag)
    if not best_lags:
        return {"cadence": cad, "error": "no pairs"}
    bl = np.array(best_lags)
    return {"cadence": cad, "n_pairs": int(len(bl)),
            "btc_leads_frac": round(float(np.mean(bl > 0)), 3),
            "contemporaneous_frac": round(float(np.mean(bl == 0)), 3),
            "alt_leads_frac": round(float(np.mean(bl < 0)), 3),
            "median_best_lag": int(np.median(bl)),
            "note": "NATIVE-resolution lead-lag (no daily resample). best_lag>0 => BTC[t] aligns alt[t+lag] => BTC leads."}


def run_cadence(cad: str) -> dict:
    res = {}
    for name, fn in [("predictability", nonlinear_predictability), ("native_leadlag", native_leadlag)]:
        t0 = time.time()
        try:
            r = fn(cad)
        except Exception as e:
            r = {"cadence": cad, "error": f"{type(e).__name__}: {e}"}
        r["_elapsed_s"] = round(time.time() - t0, 1)
        (OUT / f"pred_{name}_{cad}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        res[name] = r
        print(f"  [{cad}] {name}: {r.get('_elapsed_s')}s "
              f"{'ERROR '+str(r.get('error')) if 'error' in r else r.get('direction_AUC_gbm', r.get('btc_leads_frac',''))}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", default="1d,4h,1h,30m,15m")
    a = ap.parse_args(argv)
    for cad in [c.strip() for c in a.cadences.split(",") if c.strip()]:
        print(f"\n##### PREDICTABILITY {cad} #####", flush=True)
        run_cadence(cad)


if __name__ == "__main__":
    raise SystemExit(main())
