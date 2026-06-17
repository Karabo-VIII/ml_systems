"""src/strat/deep2020_bestbook.py -- BLOCK L: the BEST causal 2020 book = XS selection x vol sizing (honest).

The two genuinely-useful findings: XS-momentum SELECTION (which assets) + VOL-TARGET SIZING (how much). This
combines them into the best causal in-sample 2020 book and grades it honestly. Strategies (2020 H2):
  BUYHOLD          equal-weight all, full size
  VOLTGT_BH        equal-weight all, vol-target sizing
  XS_MOM           top-3 trailing winners, full size
  XS_MOM_VOLTGT    top-3 trailing winners, vol-target sizing   (the combined 'best causal' book)
Report net / maxDD / Sharpe / OOS. HONEST FRAME: XS momentum is a bull-beta tilt with weak persistence
(Block I) -> this is the best IN-SAMPLE causal book, NOT a transferable edge (dead-list: reverses in a bear).
The vol-target component is the part that likely transfers. RWYB: python -m strat.deep2020_bestbook. No emoji.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.deep2020_xsection import _panel_df, WIN, SPLIT

LB = {"1d": 7, "4h": 42}; VW = {"1d": 14, "4h": 84}; TOPK = 3


def _stats(net, ann):
    win = net[net.index >= pd.Timestamp(WIN[0])].dropna().to_numpy()
    if len(win) < 5:
        return {}
    eq = np.cumprod(1 + win); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    sh = float(np.mean(win) / (np.std(win) + 1e-12) * np.sqrt(ann))
    oosmask = net[net.index >= pd.Timestamp(WIN[0])].index >= pd.Timestamp(SPLIT)
    oos = win[oosmask.to_numpy()] if hasattr(oosmask, "to_numpy") else win[np.array(oosmask)]
    oosn = float(np.prod(1 + oos) - 1) * 100 if len(oos) else float("nan")
    return {"net": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(dd, 1), "sharpe": round(sh, 2), "oos_net": round(oosn, 1)}


def main() -> int:
    out = {}
    for cad in ["1d", "4h"]:
        ann = {"1d": 365, "4h": 365 * 6}[cad]
        df = _panel_df(cad); ret = df.pct_change(); trail = df.pct_change(LB[cad])
        rv = ret.rolling(VW[cad]).std(); med = float(rv.median().median())
        idx = df.index
        recs = {k: [] for k in ["BUYHOLD", "VOLTGT_BH", "XS_MOM", "XS_MOM_VOLTGT"]}
        times = []
        for i in range(max(LB[cad], VW[cad]) + 1, len(idx)):
            r = ret.iloc[i]; tr = trail.iloc[i - 1].dropna(); v = rv.iloc[i - 1]
            if len(tr) < TOPK + 1:
                for k in recs: recs[k].append(0.0)
                times.append(idx[i]); continue
            allsel = tr.index
            momsel = tr.nlargest(TOPK).index
            def w_vol(sel):
                w = (med / (v[sel] + 1e-12)).clip(0, 1)
                return float((w * r[sel]).sum() / max(1, len(sel)))   # avg over selected, vol-scaled
            recs["BUYHOLD"].append(float(r[allsel].mean()))
            recs["VOLTGT_BH"].append(w_vol(allsel))
            recs["XS_MOM"].append(float(r[momsel].mean()))
            recs["XS_MOM_VOLTGT"].append(w_vol(momsel))
            times.append(idx[i])
        print(f"########## {cad} -- BUYHOLD / VOLTGT_BH / XS_MOM / XS_MOM_VOLTGT (2020 H2) ##########")
        print(f"   {'book':16} {'net%':>9} {'maxDD%':>8} {'Sharpe':>7} {'OOSnet%':>8}")
        for k in ["BUYHOLD", "VOLTGT_BH", "XS_MOM", "XS_MOM_VOLTGT"]:
            s = _stats(pd.Series(recs[k], index=times), ann); out[(cad, k)] = s
            if s:
                print(f"   {k:16} {s['net']:>9} {s['maxdd']:>8} {s['sharpe']:>7} {s['oos_net']:>8}")
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump({f"{c}|{k}": v for (c, k), v in out.items()}, open(op / "bestbook.json", "w"), indent=1, default=str)
    print("HONEST: XS_MOM_VOLTGT is the best IN-SAMPLE causal book; the XS component is bull-beta (weak persistence,"
          " dead-list non-transfer), the vol-target component is the part that likely transfers.")
    print(f"[json] {op / 'bestbook.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
