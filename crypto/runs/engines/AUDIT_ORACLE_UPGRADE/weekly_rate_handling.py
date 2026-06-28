"""HOW ARE WE HANDLING THE 1-4%/wk EXPECTATION? -- honest weekly-rate distribution, by regime + static/dynamic.

User (2026-06-28): target = 1-4% per WEEK; the system (static MA + dynamic layer) should be enough to do it.
Measure where we ACTUALLY are vs that band, don't assert. For the deployable (momentum top-8 mover-selection,
breadth-gated, u100, 1d, 14d/7d, de-leaked, prod), on the clean OOS (>=2024-05-15):
  - non-overlapping 7d (weekly) return distribution: % in [1,4]%, % >4%, % >0, median, mean
  - REGIME SPLIT: weeks the gate was DEPLOYED (in-market) vs CASH (gate off) -- the dynamic risk-off
  - STATIC vs DYNAMIC: static band EW (de-risked) vs dynamic momentum mover-selection weekly rate
  - per-year (bull 2024-H2 vs bear 2025-2026) so the regime-dependence is explicit
The honest reconciliation: long-only can hit 1-4%/wk in UP-regimes but not in DOWN-weeks (the North-Star ceiling);
the dynamic gate's down-regime answer is CASH (~0%/wk), not +1-4%. Shows the lever to lift the cycle-average.
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

P.FWD_START = "2024-05-15"        # clean OOS only
BPD = P.BARS_PER_DAY[P.TF]; LB = P.LB_D*BPD; CAP = 0.33; K = 8; SB = 7*BPD


def main():
    f = P.build()
    elig = (f["conv"] > 0) & (f["g_brd"][:, None] > 0)
    vol, ret, mask, idx, cdf, A, T = f["vol"], f["roo"], f["mask"], f["idx"], f["cdf"], f["A"], len(f["idx"])
    mom = cdf.pct_change(LB).to_numpy()

    def net_from_W(W):
        Wl = np.vstack([np.zeros((1, A)), W[:-1]]); gr = np.nansum(Wl*ret, axis=1)
        turn = np.nansum(np.abs(W - Wl), axis=1); return gr - turn*(COST_PROD/2), Wl

    def W_mom(wt="invvol"):
        W = np.zeros((T, A)); s = np.where(elig & np.isfinite(mom), mom, -np.inf)
        for t in range(T):
            e = np.where(np.isfinite(s[t]))[0]
            if len(e) == 0: continue
            kk = min(K, len(e)); top = e[np.argsort(-s[t][e])[:kk]]
            v = vol[t][top]; w = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0) if wt == "invvol" else np.ones(kk)
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

    def weekly(net, Wl):
        """non-overlapping 7d slice returns (%) + per-slice deployed fraction."""
        nm = net[mask]; gross = Wl.sum(axis=1)[mask]; dts = pd.DatetimeIndex(idx[mask])
        wk, dep, wdt = [], [], []
        i = 0
        while i+SB <= len(nm):
            wk.append(float(np.prod(1+nm[i:i+SB])-1)*100)
            dep.append(float(np.mean(gross[i:i+SB] > 1e-9)))
            wdt.append(dts[i]); i += SB
        return np.array(wk), np.array(dep), pd.DatetimeIndex(wdt)

    def dist(wk):
        return {"n": len(wk), "mean": round(float(np.mean(wk)), 2), "median": round(float(np.median(wk)), 2),
                "pct_in_1to4": round(float(np.mean((wk >= 1) & (wk <= 4))), 3),
                "pct_ge1": round(float(np.mean(wk >= 1)), 3), "pct_gt4": round(float(np.mean(wk > 4)), 3),
                "pct_pos": round(float(np.mean(wk > 0)), 3), "p10": round(float(np.percentile(wk, 10)), 2),
                "p90": round(float(np.percentile(wk, 90)), 2)}

    out = {"check": "WEEKLY_RATE_HANDLING", "target": "1-4% per week", "window": "clean OOS >=2024-05-15",
           "ts": datetime.now().strftime("%Y%m%d_%H%M%S"), "deployable_mom_top8_invvol": {}, "static_derisk": {}}
    print("="*94)
    print("WEEKLY (7d) RETURN vs the 1-4%/wk TARGET  -- deployable = momentum top-8, clean OOS (>=2024-05-15)")
    print("="*94)
    for nm, W in [("DYNAMIC: momentum top8", W_mom("invvol")), ("STATIC: de-risk band EW", W_derisk())]:
        net, Wl = net_from_W(W); wk, dep, wdt = weekly(net, Wl)
        d = dist(wk)
        # regime split: deployed weeks (dep>0.5) vs cash-ish weeks (dep<0.5)
        inwk = wk[dep > 0.5]; cashwk = wk[dep <= 0.5]
        # per-year
        yr = {}
        for y in sorted(set(wdt.year)):
            m = wdt.year == y
            if m.sum(): yr[str(y)] = {"mean_wk": round(float(np.mean(wk[m])), 2), "pct_in_1to4": round(float(np.mean((wk[m] >= 1) & (wk[m] <= 4))), 2), "n": int(m.sum())}
        key = "deployable_mom_top8_invvol" if "DYNAMIC" in nm else "static_derisk"
        out[key] = {"all": d, "deployed_weeks": dist(inwk) if len(inwk) else None,
                    "cash_weeks": dist(cashwk) if len(cashwk) else None, "per_year": yr,
                    "deployed_frac_of_weeks": round(float(np.mean(dep > 0.5)), 3)}
        print(f"\n  {nm}:  mean {d['mean']:+.2f}%/wk  median {d['median']:+.2f}%  "
              f"| weeks in[1,4%]={d['pct_in_1to4']:.0%}  >=1%={d['pct_ge1']:.0%}  >4%={d['pct_gt4']:.0%}  >0={d['pct_pos']:.0%}")
        if len(inwk): print(f"     DEPLOYED weeks ({np.mean(dep>0.5):.0%} of all): mean {np.mean(inwk):+.2f}%/wk  in[1,4]={np.mean((inwk>=1)&(inwk<=4)):.0%}  >=1%={np.mean(inwk>=1):.0%}")
        if len(cashwk): print(f"     CASH-ish weeks ({np.mean(dep<=0.5):.0%}): mean {np.mean(cashwk):+.2f}%/wk")
        print(f"     per-year: " + " | ".join(f"{y}: {v['mean_wk']:+.2f}%/wk in[1,4]={v['pct_in_1to4']:.0%}" for y,v in yr.items()))
    with open(HERE/"weekly_rate_handling.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'weekly_rate_handling.json'}")


if __name__ == "__main__":
    main()
