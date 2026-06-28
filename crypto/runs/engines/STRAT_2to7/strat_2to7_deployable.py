"""STRAT 2-7%/wk -- the campaign's DEPLOYABLE: momentum top-5 + TAKE-PROFIT 5% + breadth gate.

The 6-cycle campaign's one genuine win for the 2-7%/wk OBJECTIVE: a TP~5% cap on the momentum top-5 book raises
the % of deployed weeks landing in [2,7] (the band) cross-window, at a compound cost. This ships it: the deployed-
week distribution (the honest "2-7%/wk reality"), the full-OOS $ track, and the LIVE weekly candidate book.
Spec: u100, 1d, 14d/7d rolling, breadth-gated, momentum top-5 inv-vol, TP5, de-leaked, prod, next-open fills, $10k.
HONEST: 2-7%/wk is a CONDITIONAL reality -- real when deployed in favorable regimes (~25% of weeks in band, median
+3-7%/wk in bulls); CASH in bears (long-only can't green a down-week). TP5 trades the >7% tail for band-consistency.
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

BPD = M.BPD; K = M.K; SL = M.SL; TP = 0.05


def main():
    f = M.build("FWD")            # full history through ~2026-05; we slice OOS below
    idx = f["idx"]; cdfcols = list(f["cdf"].columns)
    net, Wl = E.book_with_exits(f, f["scores"]["mom14"], sl=None, tp=TP, trail=None)
    W = E.book_with_exits._lastW                       # unlagged target weights
    oos = (idx >= pd.Timestamp("2024-05-15"))
    d = E.wdist(net, Wl, oos)                          # deployed-week distribution on full OOS
    no = net[oos]; eq = 10000*np.cumprod(1+no)
    dd = float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100
    last = f["T"]-1; livew = W[last]                   # live book = target entering the most recent bar
    liveheld = sorted(((cdfcols[j], round(float(livew[j]),3)) for j in np.where(livew>1e-6)[0]), key=lambda x:-x[1])
    out = {"check":"STRAT_2to7_DEPLOYABLE","spec":"u100 1d 14d/7d momentum top5 inv-vol + TP5, breadth-gated, de-leaked, prod",
           "ts":datetime.now().strftime("%Y%m%d_%H%M%S"),
           "oos_deployed_week_dist": d, "oos_final_equity": round(float(eq[-1]),0),
           "oos_compound_pct": round(float(eq[-1]/10000-1)*100,1), "oos_maxdd_pct": round(dd,1),
           "live_as_of": str(pd.Timestamp(idx[last]).date()), "live_n": len(liveheld), "live_candidates": liveheld}
    print("="*88)
    print("DEPLOYABLE: momentum top-5 + TP5 + breadth gate  (u100, 1d, 14d/7d, de-leaked, prod, $10k)")
    print("="*88)
    print(f"  FULL OOS (2024-05-15..{str(pd.Timestamp(idx[last]).date())}): ${eq[-1]:,.0f}  compound {out['oos_compound_pct']:+.0f}%  maxDD {dd:+.0f}%")
    print(f"  DEPLOYED-WEEK distribution: dep_wks={d['dep']}  median {d['median']:+.2f}%/wk  mean {d['mean']:+.2f}%  "
          f"hit>=2 {d['hit2']:.0%}  in[2,7] {d['in27']:.0%}  neg {d['neg']:.0%}")
    print(f"\n  LIVE candidate book as of {out['live_as_of']} ({'DEPLOY' if liveheld else 'CASH'}, {len(liveheld)} names):")
    if liveheld:
        for n,w in liveheld: print(f"      {n:<14} {w:6.1%}")
    else:
        print("      [CASH] breadth gate OFF -- hold cash (no candidates qualify this week)")
    with open(HERE/"strat_2to7_deployable.json","w") as fo: json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'strat_2to7_deployable.json'}")


if __name__ == "__main__":
    main()
