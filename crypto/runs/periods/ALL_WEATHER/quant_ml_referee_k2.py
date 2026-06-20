"""
quant_ml_referee_k2.py -- INDEPENDENT DERIVATION #2 (different code path, different estimator/sampler).

Purpose: confirm derivation #1's strict-walk-forward number is NOT an artifact of my own harness.
Changes vs #1 (deliberately orthogonal implementation):
  - estimator: LogisticRegression on PER-FOLD-standardized features (scaler fit on train fold ONLY)
  - walk-forward written from scratch with an EXPLICIT embargo (drop last 2*H train bars before cutoff)
  - slice sampler: random start bars but evaluate ABS win-rate (slice fwd>0) AND a 1000-draw resample
  - leak-twin here = fit the scaler on the FULL sample (the classic 'global standardization' leak,
    G-AUDIT-011 class) while keeping the model walk-forward -> isolates the scaler-leak channel alone.

No emoji. Does not commit.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(r"c:\Users\karab\Documents\coding\ml_systems\crypto\src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import strat.mover_lab as lab
from sklearn.linear_model import LogisticRegression

SEED = 7
H = 7

FEATS = ["dist_sma200", "dist_sma50", "range_pos", "rsi14", "mom7", "mom14", "mom30",
         "ret1", "ret3", "mom14_rank", "gate", "breadth", "btc_regime"]


def panels(ind):
    C = ind["C"]; eps = 1e-8
    P = {
        "dist_sma200": C / (ind["sma200"] + eps) - 1,
        "dist_sma50":  C / (ind["sma50"] + eps) - 1,
        "range_pos":   (C - ind["ll14"]) / (ind["hh14"] - ind["ll14"] + eps),
        "rsi14":       ind["rsi14"],
        "mom7": ind["mom7"], "mom14": ind["mom14"], "mom30": ind["mom30"],
        "ret1": ind["ret1"], "ret3": C / C.shift(3) - 1,
        "mom14_rank": ind["mom14"].rank(axis=1, pct=True),
        "gate": ind["gate"].astype(float),
    }
    breadth = (C > ind["sma50"]).astype(float).mean(axis=1)
    btc = (C["BTCUSDT"] > ind["sma200"]["BTCUSDT"]).astype(float).fillna(0.0)
    P["breadth"] = pd.DataFrame({s: breadth for s in C.columns})
    P["btc_regime"] = pd.DataFrame({s: btc for s in C.columns})
    fwd = C.shift(-H) / C - 1
    return P, fwd, C


def run(global_scaler_leak=False):
    ind = lab.load("2020-01-01", "2026-06-01")
    P, fwd, C = panels(ind)
    nD, nA = C.shape
    X = np.column_stack([P[k].to_numpy().reshape(-1) for k in FEATS])
    yraw = fwd.to_numpy().reshape(-1)
    y = (yraw > 0).astype(float)
    di = np.repeat(np.arange(nD), nA)
    vl = ~np.isnan(yraw); vf = ~np.isnan(X).any(axis=1)

    # global-scaler-leak twin: fit mean/std on ALL valid rows (future included)
    if global_scaler_leak:
        m = np.nanmean(X[vf], axis=0); s = np.nanstd(X[vf], axis=0) + 1e-8

    prob = np.full(X.shape[0], np.nan)
    retrain = 60; min_train = 600; embargo = 2 * H
    start_i = None
    for i in range(nD):
        adm = vl & vf & (di <= i - H - 1 - embargo)
        if adm.sum() >= min_train:
            start_i = i; break
    model = None; sc_m = None; sc_s = None; last = -10**9
    for i in range(start_i, nD):
        if i - last >= retrain:
            adm = vl & vf & (di <= i - H - 1 - embargo)
            Xt, yt = X[adm], y[adm]
            if len(Xt) >= min_train:
                if global_scaler_leak:
                    sc_m, sc_s = m, s            # LEAK: scaler from full sample
                else:
                    sc_m = Xt.mean(axis=0); sc_s = Xt.std(axis=0) + 1e-8  # honest: train-fold only
                model = LogisticRegression(C=0.2, max_iter=500, random_state=SEED)
                model.fit((Xt - sc_m) / sc_s, yt)
                last = i
        if model is None:
            continue
        day = (di == i) & vf
        if day.any():
            prob[day] = model.predict_proba((X[day] - sc_m) / sc_s)[:, 1]
    return prob, di, C, start_i


def slice_wr(prob, di, C, start_i, topk=5, thr=0.5, n=400, seed=SEED):
    rng = np.random.default_rng(seed)
    Cv = C.to_numpy(); nD, nA = C.shape
    probM = prob.reshape(nD, nA)
    starts = np.arange(start_i, nD - H - 1)
    pick = rng.choice(starts, size=n, replace=True)
    ml, bh = [], []
    for s in pick:
        fwd_row = Cv[s + H] / Cv[s] - 1.0
        listed = ~np.isnan(fwd_row)
        if not listed.any():
            continue
        bh.append(np.nanmean(fwd_row[listed]))
        pr = probM[s].copy(); pr[~listed] = -np.inf
        elig = np.where((pr >= thr) & np.isfinite(pr))[0]
        if elig.size == 0:
            ml.append(0.0); continue
        order = elig[np.argsort(-pr[elig])][:topk]
        ml.append(float(np.mean(fwd_row[order])))
    ml, bh = np.array(ml), np.array(bh)
    return dict(ml_wr=float((ml > 0).mean()), bh_wr=float((bh > 0).mean()),
               ml_mean=float(ml.mean()), bh_mean=float(bh.mean()), n=len(ml))


if __name__ == "__main__":
    print("DERIVATION #2 (LogReg, per-fold scaler, embargo=2H) -- independent re-derivation")
    for leak in (False, True):
        prob, di, C, si = run(global_scaler_leak=leak)
        best = None
        for k in (3, 5, 10):
            for t in (0.5, 0.45):
                r = slice_wr(prob, di, C, si, topk=k, thr=t)
                tag = "GLOBAL-SCALER-LEAK" if leak else "STRICT"
                print(f"  {tag:<20} K={k} thr={t:.2f}: ml_WR={r['ml_wr']*100:.1f}% "
                      f"bh_WR={r['bh_wr']*100:.1f}% ml_mean={r['ml_mean']*100:+.2f}% "
                      f"bh_mean={r['bh_mean']*100:+.2f}% n={r['n']}")
