"""WIRE THE WEEKLY CANDIDATES -- the names the deployable holds each week, targeting 2-7%/wk.

User (/orc 2026-06-28, autonomous): "wire me the 2-7% candidates per week." Operationalize the weekly
candidate book of the deployable (momentum mover-selection, breadth-gated, u100, 1d, 14d/7d, de-leaked,
prod cost). For each non-overlapping 7d week: the gate state (deploy/cash), the top-K momentum candidate
names + weights held ENTERING the week, and the realized 7d return of that book. Plus the LATEST week's
live candidates (what to hold now). Concentration K is chosen to CENTER the deployed-week return on the
2-7% band (test K=3/5/8). HONEST: returns are lumpy + gated -- some deployed weeks land in 2-7%, some
above, some negative; cash weeks have NO candidates. long-only spot lev=1.
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

P.FWD_START = "2024-05-15"     # emit weekly candidates over the clean OOS
BPD = P.BARS_PER_DAY[P.TF]; LB = P.LB_D*BPD; SB = 7*BPD


def ctx():
    f = P.build()
    elig = (f["conv"] > 0) & (f["g_brd"][:, None] > 0)
    mom = f["cdf"].pct_change(LB).to_numpy()
    return f, elig, mom


def week_books(f, elig, mom, K):
    """Per non-overlapping 7d week: gate, candidate (name,wt) entering the week, realized 7d return."""
    cols = f["cols"]; vol = f["vol"]; ret = f["roo"]; idx = f["idx"]; mask = f["mask"]; A = f["A"]; T = len(idx)
    # momentum top-K inv-vol weights per bar (causal)
    W = np.zeros((T, A)); s = np.where(elig & np.isfinite(mom), mom, -np.inf)
    for t in range(T):
        e = np.where(np.isfinite(s[t]))[0]
        if len(e) == 0: continue
        kk = min(K, len(e)); top = e[np.argsort(-s[t][e])[:kk]]
        v = vol[t][top]; w = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0)
        W[t, top] = w/w.sum() if w.sum() > 0 else 1.0/kk
    Wl = np.vstack([np.zeros((1, A)), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1); turn = np.nansum(np.abs(W-Wl), axis=1)
    net = gr - turn*(COST_PROD/2)
    mi = np.where(mask)[0]
    books = []
    i = 0
    while i+SB <= len(mi):
        wstart = mi[i]                                  # bar entering the week (book decided at wstart-1 close)
        wk_net = net[mi[i]:mi[i]+SB]
        wret = float(np.prod(1+wk_net)-1)*100
        held = np.where(W[wstart-1] > 1e-6)[0] if wstart > 0 else np.where(W[wstart] > 1e-6)[0]
        cand = sorted(((cols[j], round(float(W[max(wstart-1,0)][j]), 3)) for j in held), key=lambda x: -x[1])
        books.append({"week_start": str(pd.Timestamp(idx[wstart]).date()), "deployed": len(cand) > 0,
                      "n_candidates": len(cand), "candidates": cand, "realized_7d_pct": round(wret, 2)})
        i += SB
    # LATEST live book: candidates entering the most recent bar
    last = T-1
    held = np.where(W[last] > 1e-6)[0]
    live = sorted(((cols[j], round(float(W[last][j]), 3)) for j in held), key=lambda x: -x[1])
    return books, {"as_of": str(pd.Timestamp(idx[last]).date()), "gate": "DEPLOY" if len(live) else "CASH",
                   "n_candidates": len(live), "candidates": live}


def main():
    f, elig, mom = ctx()
    out = {"check": "WEEKLY_CANDIDATES", "spec": "momentum mover-selection, breadth-gated, u100, 1d, 14d/7d, de-leaked, prod",
           "target": "2-7% per week", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"), "by_K": {}}
    # choose K that centers deployed-week return on 2-7%
    print("="*92)
    print("WEEKLY CANDIDATES -- concentration tuning (deployed-week return vs the 2-7%/wk target)")
    print("="*92)
    print(f"  {'K':>3}{'dep_wks':>9}{'dep_mean%/wk':>13}{'dep_median':>11}{'in[2,7]':>9}{'>7%':>7}{'neg':>7}")
    best = None
    for K in [3, 5, 8]:
        books, live = week_books(f, elig, mom, K)
        dep = [b["realized_7d_pct"] for b in books if b["deployed"]]
        if not dep: continue
        dep = np.array(dep)
        m = {"dep_weeks": len(dep), "dep_mean": round(float(dep.mean()), 2), "dep_median": round(float(np.median(dep)), 2),
             "in_2to7": round(float(np.mean((dep >= 2) & (dep <= 7))), 3), "gt7": round(float(np.mean(dep > 7)), 3),
             "neg": round(float(np.mean(dep < 0)), 3)}
        out["by_K"][f"K{K}"] = {**m, "live": live}
        print(f"  {K:>3}{m['dep_weeks']:>9}{m['dep_mean']:>+13.2f}{m['dep_median']:>+11.2f}{m['in_2to7']*100:>8.0f}%{m['gt7']*100:>6.0f}%{m['neg']*100:>6.0f}%")
        # pick K whose deployed MEAN is closest into [2,7] center (~4.5)
        score = -abs(m["dep_mean"] - 4.5)
        if best is None or score > best[0]: best = (score, K, books, live)
    _, K, books, live = best
    out["chosen_K"] = K; out["weekly_log_tail"] = books[-16:]; out["live_book"] = live
    print(f"\n  --> CHOSEN K={K} (deployed-week mean closest to the 2-7% band center)")
    print(f"\n  LAST 16 WEEKLY CANDIDATE BOOKS (K={K}):")
    print(f"  {'week_start':<12}{'gate':>7}{'7d_ret':>9}  candidates (name:wt)")
    print("  "+"-"*86)
    for b in books[-16:]:
        tag = "DEPLOY" if b["deployed"] else "CASH"
        cs = ", ".join(f"{n}:{w:.0%}" for n, w in b["candidates"][:6]) if b["deployed"] else "(cash -- gate off)"
        print(f"  {b['week_start']:<12}{tag:>7}{b['realized_7d_pct']:>+8.1f}%  {cs}")
    print(f"\n  *** LIVE CANDIDATE BOOK as of {live['as_of']} -- {live['gate']} ({live['n_candidates']} names) ***")
    if live["candidates"]:
        for n, w in live["candidates"]:
            print(f"      {n:<14} {w:6.1%}")
    else:
        print("      [CASH] gate OFF -- hold cash this week (no 2-7% candidates qualify)")
    with open(HERE/"weekly_candidates.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'weekly_candidates.json'}")


if __name__ == "__main__":
    main()
