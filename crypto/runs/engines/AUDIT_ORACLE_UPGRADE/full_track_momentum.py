"""FULL CONTINUOUS TRACK -- momentum mover-selection vs de-risked vs buy-hold, $10k, prod-grade.

Answers the user's critique end-to-end: does the MA family, configured to CATCH THE MOVERS (momentum
top-8), actually make money over time? Runs ONE continuous paper-trade from 2023-07-01 to data end
(~2026-05-28), $10k start, 1d, 14d/7d rolling, breadth-gated, de-leaked, prod cost, next-open fills.

HONESTY (RWYB): the frozen bands were SELECTED on TRAIN<=2024-05-15, so the 2023->mid-2024 chunk is
IN-SAMPLE-inflated -- this script reports the FULL track AND the OOS-only (>=2024-05-15) split so the
in-sample portion is not headlined as OOS. The RELATIVE comparison (momentum vs de-risked, same band)
is apples-to-apples regardless. long-only spot lev=1.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_trade_ma_family as P
from strat.canonical_exits import COST_PROD

P.FWD_START = "2023-07-01"      # full continuous span (VAL start -> data end)
BPD = P.BARS_PER_DAY[P.TF]; LB = P.LB_D*BPD; CAP = 0.33; K = 8
OOS_CUT = pd.Timestamp("2024-05-15")


def build_ctx():
    f = P.build()
    return f, (f["conv"] > 0) & (f["g_brd"][:, None] > 0)


def main():
    f, elig = build_ctx()
    vol, ret, mask, idx, cdf, A, T = f["vol"], f["roo"], f["mask"], f["idx"], f["cdf"], f["A"], len(f["idx"])
    mom = cdf.pct_change(LB).to_numpy()

    def net_from_W(W):
        Wl = np.vstack([np.zeros((1, A)), W[:-1]]); gr = np.nansum(Wl*ret, axis=1)
        turn = np.nansum(np.abs(W - Wl), axis=1); return gr - turn*(COST_PROD/2)

    def W_topk(score, wt):
        W = np.zeros((T, A)); s = np.where(elig & np.isfinite(score), score, -np.inf)
        for t in range(T):
            e = np.where(np.isfinite(s[t]))[0]
            if len(e) == 0: continue
            kk = min(K, len(e)); top = e[np.argsort(-s[t][e])[:kk]]
            v = vol[t][top]
            w = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0) if wt == "invvol" else np.ones(kk)
            W[t, top] = w/w.sum() if w.sum() > 0 else 1.0/kk
        return W

    def W_derisk():
        W = np.zeros((T, A))
        for t in range(T):
            e = np.where(elig[t])[0]
            if len(e) == 0: continue
            v = vol[t][e]; w = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0)
            w = np.minimum(w/w.sum() if w.sum() > 0 else 1.0/len(e), CAP)
            if w.sum() > 0: W[t, e] = w/w.sum()
        return W

    def stats(net, sub):
        n = net[mask][sub]
        if len(n) == 0: return None
        eq = 10000*np.cumprod(1+n)
        dd = float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100
        sh = (np.mean(n)*BPD*365)/(np.std(n)*np.sqrt(BPD*365)) if np.std(n) > 1e-12 else 0
        return {"final": round(float(eq[-1]), 0), "compound": round(float(eq[-1]/10000-1)*100, 1),
                "maxdd": round(dd, 1), "sharpe": round(float(sh), 2)}

    dates = pd.DatetimeIndex(idx[mask]); full = np.ones(len(dates), bool); oos = np.asarray(dates >= OOS_CUT)
    def yearly(net):
        s = pd.Series(net[mask], index=dates)
        return {str(y): round(float(np.prod(1+g.to_numpy())-1)*100, 1) for y, g in s.groupby(s.index.year)}

    schemes = {"mom top8 inv-vol": W_topk(mom, "invvol"), "mom top8 EW": W_topk(mom, "ew"),
               "inv-vol de-risk (shipped)": W_derisk()}
    out = {"check": "FULL_TRACK_MOMENTUM", "span": f"2023-07-01..{str(dates[-1].date())}",
           "note": "2023->2024-05 band is IN-SAMPLE; OOS split reported separately. RELATIVE comp is apples-to-apples.",
           "ts": datetime.now().strftime("%Y%m%d_%H%M%S"), "schemes": {}}
    print("="*98)
    print(f"FULL TRACK 2023-07-01..{str(dates[-1].date())} ({int(mask.sum())} bars) $10k | FULL (incl in-sample) + OOS-only (>=2024-05-15)")
    print("="*98)
    print(f"  {'scheme':<26}{'FULL$':>10}{'FULLcmp':>9}{'OOS$10k->':>11}{'OOScmp':>8}{'OOSdd':>7}{'OOSsh':>7}  per-year(full)")
    print("  "+"-"*92)
    for nm, W in schemes.items():
        net = net_from_W(W); F = stats(net, full); O = stats(net, oos); yr = yearly(net)
        out["schemes"][nm] = {"full": F, "oos_only": O, "yearly_full": yr}
        # OOS-only $ from 10k
        no = net[mask][oos]; oeq = 10000*np.cumprod(1+no)
        print(f"  {nm:<26}{F['final']:>10.0f}{F['compound']:>+8.0f}%{oeq[-1]:>11.0f}{O['compound']:>+7.0f}%{O['maxdd']:>+7.0f}{O['sharpe']:>+7.2f}  {yr}")
    def stats_masked(n):   # benchmarks come PRE-masked (len == len(dates))
        eq = 10000*np.cumprod(1+n)
        dd = float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100
        return {"final": round(float(eq[-1]), 0), "compound": round(float(eq[-1]/10000-1)*100, 1), "maxdd": round(dd, 1)}
    for w in ["alt", "btc"]:
        nb, _ = P.bench_net(f, w)              # already masked to the FULL span
        F = stats_masked(nb); O = stats_masked(nb[oos]); oeq = 10000*np.cumprod(1+nb[oos])
        out["schemes"][f"BH_{w}"] = {"full": F, "oos_only": O}
        print(f"  {'BH '+w:<26}{F['final']:>10.0f}{F['compound']:>+8.0f}%{oeq[-1]:>11.0f}{O['compound']:>+7.0f}%{O['maxdd']:>+7.0f}")
    with open(HERE/"full_track_momentum.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'full_track_momentum.json'}")


if __name__ == "__main__":
    main()
