"""STRAT 2-7%/wk -- Cycle 4: BEAR-DETECTOR regime composite (stay CASH in the 2025-26 downturn).

Cycle 3: the bull is largely solvable; the BEAR (FWD) is the ceiling -- the breadth gate deploys ~30% into red
chop. Test stronger deploy CONDITIONS to keep the system in cash during a falling market, WITHOUT hurting bulls:
  breadth (base) | btc_trend (BTC>SMA100, the V1 gate) | breadth&btc | breadth&not_mktDD (EW index not >X% below
  60d high) | breadth&btc&not_mktDD. Each x momentum top-K + TP7. Measure FWD deploy-fraction + deployed-week median
  (want: FWD deploy DOWN + median UP) WITHOUT degrading VAL/OOS24. The robust winner cuts bear-deploys, keeps bulls.
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

BPD = M.BPD; K = M.K; SW = M.SW


def hyst_arr(sig, up, dn):
    n = len(sig); s = np.zeros(n, int); st = 0
    up = np.full(n, up, float) if np.isscalar(up) else np.asarray(up, float)
    dn = np.full(n, dn, float) if np.isscalar(dn) else np.asarray(dn, float)
    for i in range(n):
        if not (np.isfinite(sig[i]) and np.isfinite(up[i]) and np.isfinite(dn[i])): s[i] = st; continue
        if st == 0 and sig[i] > up[i]: st = 1
        elif st == 1 and sig[i] < dn[i]: st = 0
        s[i] = st
    return s


def gates(f):
    cdf = f["cdf"]; brd = f["brd"]; T = f["T"]; C = cdf.to_numpy()
    g_brd = M.hyst(brd, 0.5, 0.4)
    # BTC trend gate (BTC > SMA100, hysteresis 1.02/0.98)
    btc = "BTCUSDT" if "BTCUSDT" in cdf.columns else cdf.columns[0]
    bc = cdf[btc]; bsma = bc.rolling(SW, min_periods=SW//2).mean().to_numpy()
    g_btc = hyst_arr(bc.to_numpy(), bsma*1.02, bsma*0.98)
    # market index (EW of all live names) drawdown-from-60d-high gate: deploy only if DD > -12%
    idxlvl = pd.Series(np.nanmean(cdf.pct_change().fillna(0).to_numpy(), axis=1)).add(1).cumprod().to_numpy()
    roll = pd.Series(idxlvl).rolling(60*BPD, min_periods=20).max().to_numpy()
    mdd = idxlvl/np.where(roll > 0, roll, np.nan) - 1.0
    g_notdd = (np.nan_to_num(mdd, nan=0.0) > -0.12).astype(int)
    return {"breadth": g_brd, "btc_trend": g_btc, "breadth&btc": (g_brd & g_btc).astype(int),
            "breadth&notDD": (g_brd & g_notdd).astype(int), "breadth&btc&notDD": (g_brd & g_btc & g_notdd).astype(int)}


def main():
    F = {w: M.build(w) for w in ["VAL","OOS24","FWD"]}
    G = {w: gates(F[w]) for w in F}
    names = ["breadth","btc_trend","breadth&btc","breadth&notDD","breadth&btc&notDD"]
    out = {"check":"STRAT_2to7_cycle4_regime","score":"mom14","exit":"TP7","K":K,
           "ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for w in ["VAL","OOS24","FWD"]:
        f = F[w]; out["windows"][w] = {}
        print(f"\n{'='*98}\n{w}  BEAR-DETECTOR gates x mom14 top-{K} + TP7 (want FWD deploy DOWN + median UP)")
        print(f"{'='*98}")
        print(f"  {'gate':<18}{'dep_frac':>9}{'dep_wks':>8}{'median':>8}{'mean':>7}{'hit>=2':>8}{'in[2,7]':>9}{'neg':>6}{'comp':>8}")
        print("  "+"-"*84)
        for nm in names:
            g = G[w][nm]; f2 = dict(f); f2["elig"] = (f["conv"] > 0) & (g[:, None] > 0)
            net, Wl = E.book_with_exits(f2, f2["scores"]["mom14"], sl=None, tp=0.07, trail=None)
            d = E.wdist(net, Wl, f2["mask"]); df = round(float(np.mean(Wl.sum(axis=1)[f2["mask"]] > 1e-9)), 3)
            if d is None:
                print(f"  {nm:<18}{df*100:>8.0f}%   (no deployed weeks)"); out["windows"][w][nm]={"deploy_frac":df}; continue
            out["windows"][w][nm] = {**d, "deploy_frac": df}
            print(f"  {nm:<18}{df*100:>8.0f}%{d['dep']:>8}{d['median']:>+8.2f}{d['mean']:>+7.2f}{d['hit2']*100:>7.0f}%{d['in27']*100:>8.0f}%{d['neg']*100:>5.0f}%{d['comp']:>+7.0f}%")
    with open(HERE/"strat_2to7_regime.json","w") as fo: json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'strat_2to7_regime.json'}")


if __name__ == "__main__":
    main()
