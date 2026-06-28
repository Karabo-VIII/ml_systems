"""STRAT 2-7%/wk -- Cycle 3: deploy-SELECTIVITY (cut the bear-deploy red weeks) x take-profit.

Cycle 2: TP raises in-band hit but the bear (FWD) still deploys into red weeks. The deployed-week distribution
conditions on WHICH weeks are deployed -- a sharper gate that only deploys in favorable regimes should cut the
bear red weeks and lift the deployed median. Test gate variants (breadth on/off threshold + rising-breadth) x
TP7, measure deployed-week dist + DEPLOY-FRACTION (how often in-market) + compound, cross-window. Reuses cycle-1
build() + cycle-2 book_with_exits().
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

BPD = M.BPD; K = M.K; SL = M.SL


def regate(f, up, dn, rising_k=0):
    brd = f["brd"]; g = M.hyst(brd, up, dn)
    if rising_k > 0:
        rise = np.zeros(len(brd), int)
        for t in range(rising_k, len(brd)):
            rise[t] = 1 if (np.isfinite(brd[t]) and np.isfinite(brd[t-rising_k]) and brd[t] >= brd[t-rising_k]) else 0
        g = (g & rise).astype(int)
    f2 = dict(f); f2["elig"] = (f["conv"] > 0) & (g[:, None] > 0)
    return f2


def deploy_frac(Wl, mask):
    g = Wl.sum(axis=1)[mask]; return round(float(np.mean(g > 1e-9)), 3)


def main():
    F = {w: M.build(w) for w in ["VAL","OOS24","FWD"]}
    gates = [("base .5/.4", .5, .4, 0), ("hi .6/.45", .6, .45, 0), ("hi .7/.5", .7, .5, 0),
             ("rise .5/.4", .5, .4, 7), ("hi+rise .6/.45", .6, .45, 7)]
    out = {"check":"STRAT_2to7_cycle3_deploy","score":"mom14","exit":"TP7","K":K,
           "ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for w in ["VAL","OOS24","FWD"]:
        out["windows"][w] = {}
        print(f"\n{'='*98}\n{w}  DEPLOY-SELECTIVITY x TP7 on mom14 top-{K} (deployed-week dist + deploy-fraction)")
        print(f"{'='*98}")
        print(f"  {'gate':<16}{'dep_frac':>9}{'dep_wks':>8}{'median':>8}{'mean':>7}{'hit>=2':>8}{'in[2,7]':>9}{'neg':>6}{'comp':>8}")
        print("  "+"-"*82)
        for nm, up, dn, rk in gates:
            f2 = regate(F[w], up, dn, rk)
            net, Wl = E.book_with_exits(f2, f2["scores"]["mom14"], sl=None, tp=0.07, trail=None)
            d = E.wdist(net, Wl, f2["mask"]); df = deploy_frac(Wl, f2["mask"])
            if d is None:
                print(f"  {nm:<16}{df:>9.0%}  (no deployed weeks)"); continue
            out["windows"][w][nm] = {**d, "deploy_frac": df}
            print(f"  {nm:<16}{df*100:>8.0f}%{d['dep']:>8}{d['median']:>+8.2f}{d['mean']:>+7.2f}{d['hit2']*100:>7.0f}%{d['in27']*100:>8.0f}%{d['neg']*100:>5.0f}%{d['comp']:>+7.0f}%")
    with open(HERE/"strat_2to7_deploy.json","w") as fo: json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'strat_2to7_deploy.json'}")


if __name__ == "__main__":
    main()
