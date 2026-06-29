"""STRAT 2-7%/wk -- Cycle 5: LEARNED META-LABELER -- does P(top-mover next 7d | features) beat momentum?

The SOTA move: learn which names will be top movers from a multi-factor causal feature set, vs the single
momentum rule. Features (causal): mom14, accel, breakout, volexp, relrank. Label: name is a top-K forward-7d
mover among the eligible set at that bar. TRAIN on <=2024-05-15 (TRAIN+VAL), TEST on >2024-05-15 (OOS24+FWD).
GBM + logistic; rank eligible by P, top-K, +TP7; deployed-week dist on OOS24 + FWD vs mom14+TP7 baseline + a
random-P null. If the learned predictor does NOT beat momentum, the ceiling is the MARKET, not the predictor.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strat_2to7_movers as M
import strat_2to7_exits as E
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

BPD = M.BPD; K = M.K; SL = M.SL; CUT = pd.Timestamp("2024-05-15")
FEATS = ["mom14","accel","breakout","volexp","relrank"]


def main():
    f = M.build("FWD")     # full-history arrays (wall 2026-07-01)
    idx = f["idx"]; elig = f["elig"]; roo = f["roo"]; scores = f["scores"]; T = f["T"]; A = f["A"]
    # forward-7d return per name (for the LABEL only)
    fwd = np.full((T, A), np.nan)
    for t in range(T-SL):
        fwd[t] = np.prod(1+roo[t:t+SL], axis=0) - 1.0
    # label: top-K forward mover among eligible at t
    lab = np.zeros((T, A), np.int8)
    for t in range(T-SL):
        e = np.where(elig[t] & np.isfinite(fwd[t]))[0]
        if len(e) == 0: continue
        kk = min(K, len(e)); top = e[np.argsort(-fwd[t][e])[:kk]]; lab[t, top] = 1
    Fm = np.stack([np.nan_to_num(scores[k], nan=0.0) for k in FEATS], axis=2)  # (T,A,nf)
    train_t = np.array([t for t in range(T-SL) if idx[t] < CUT])
    rows_X, rows_y = [], []
    for t in train_t:
        e = np.where(elig[t])[0]
        if len(e) == 0: continue
        rows_X.append(Fm[t][e]); rows_y.append(lab[t][e])
    X = np.vstack(rows_X); y = np.concatenate(rows_y)
    out = {"check":"STRAT_2to7_cycle5_metalabeler","feats":FEATS,"K":K,"n_train_rows":int(len(y)),
           "base_rate":round(float(y.mean()),3),"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"models":{}}
    print(f"meta-labeler: train rows={len(y)}, base_rate(top-mover)={y.mean():.3f}")
    models = {"GBM": GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.7, random_state=0),
              "logit": LogisticRegression(max_iter=500, C=1.0)}
    Pmats = {}
    for nm, mdl in models.items():
        mdl.fit(X, y)
        P = np.full((T, A), -np.inf)
        for t in range(T):
            e = np.where(elig[t])[0]
            if len(e) == 0: continue
            P[t][e] = mdl.predict_proba(Fm[t][e])[:, 1]
        Pmats[nm] = P
    # windows
    oos24 = (idx >= CUT) & (idx < pd.Timestamp("2025-01-01"))
    fwdw = (idx >= pd.Timestamp("2025-04-01")) & (idx < pd.Timestamp("2026-07-01"))
    rng = np.random.default_rng(0)
    Prand = np.where(elig, rng.random((T, A)), -np.inf)
    cands = {"GBM": Pmats["GBM"], "logit": Pmats["logit"], "mom14 (baseline)": scores["mom14"], "random": Prand}
    for wnm, wm in [("OOS24", oos24), ("FWD", fwdw)]:
        out["models"][wnm] = {}
        print(f"\n{'='*90}\n{wnm}  meta-labeler vs momentum vs random (top-{K} + TP7, deployed-week dist)")
        print(f"{'='*90}")
        print(f"  {'selector':<18}{'dep':>5}{'median':>8}{'mean':>7}{'hit>=2':>8}{'in[2,7]':>9}{'neg':>6}{'comp':>8}")
        print("  "+"-"*72)
        for sn, P in cands.items():
            net, Wl = E.book_with_exits(f, P, sl=None, tp=0.07, trail=None)
            d = E.wdist(net, Wl, wm)
            if d is None: continue
            out["models"][wnm][sn] = d
            print(f"  {sn:<18}{d['dep']:>5}{d['median']:>+8.2f}{d['mean']:>+7.2f}{d['hit2']*100:>7.0f}%{d['in27']*100:>8.0f}%{d['neg']*100:>5.0f}%{d['comp']:>+7.0f}%")
    with open(HERE/"strat_2to7_metalabeler.json","w") as fo: json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'strat_2to7_metalabeler.json'}")


if __name__ == "__main__":
    main()
