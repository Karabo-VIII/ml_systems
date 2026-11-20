"""src/strat/deep2020_xsection_rigor.py -- BLOCK H2: is the cross-sectional edge real or cost/concentration?

Block H found XS momentum +317% vs +122% EW (in-sample 2020 H2). I flagged 3 caveats; this checks them:
  COST       -- charge turnover (assets entering/leaving the top-K) at taker (24bp) AND maker (6bp) rt;
  CONCENTRATION -- which assets does XS_MOM actually hold? is the edge 1-2 pump assets?
  ROBUSTNESS -- top-K in {1,2,3,5} and lookback in {3,7,14} -- does the edge persist or was it one knob?
RWYB: python -m strat.deep2020_xsection_rigor. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.deep2020_xsection import _panel_df, SYMS, WIN, SPLIT
TAKER, MAKER = 0.0024, 0.0006


def _book_cost(df, lb, topk, cost):
    ret = df.pct_change().fillna(0.0); trail = df.pct_change(lb); idx = df.index
    out = []; held = []; prev = set()
    for i in range(lb + 1, len(idx)):
        r = trail.iloc[i - 1].dropna()
        if len(r) < topk + 1:
            out.append(0.0); held.append(()); continue
        sel = set(r.nlargest(topk).index)
        turn = len(sel.symmetric_difference(prev)) / max(1, topk)     # fraction of book churned
        bar = float(ret.iloc[i][list(sel)].mean()) - turn * (cost / 2.0)
        out.append(bar); held.append(tuple(sel)); prev = sel
    return pd.Series(out, index=idx[lb + 1:]), held


def _net(s):
    win = s[s.index >= pd.Timestamp(WIN[0])]
    return round(float(np.prod(1 + win.to_numpy()) - 1) * 100, 1)


def main() -> int:
    print("BLOCK H2 cross-sectional rigor (cost + concentration + robustness), 2020 H2\n")
    for cad in ["1d", "4h"]:
        df = _panel_df(cad)
        lb0 = {"1d": 7, "4h": 42}[cad]
        print(f"########## {cad} ##########")
        # COST impact
        s_free, held = _book_cost(df, lb0, 3, 0.0)
        s_mk, _ = _book_cost(df, lb0, 3, MAKER); s_tk, _ = _book_cost(df, lb0, 3, TAKER)
        print(f"   XS_MOM top3 trailing-{lb0}: net free {_net(s_free)}%  maker {_net(s_mk)}%  TAKER {_net(s_tk)}%")
        # CONCENTRATION: holding frequency per asset
        cnt = Counter()
        for h in held:
            for sym in h:
                cnt[sym] += 1
        tot = sum(cnt.values())
        topc = ", ".join(f"{s.replace('USDT','')}:{round(100*n/tot)}%" for s, n in cnt.most_common(5))
        print(f"   holdings concentration (share of asset-bars held): {topc}")
        # leave-one-out: drop the top-held asset, recompute (is the edge 1 asset?)
        top_asset = cnt.most_common(1)[0][0]
        df2 = df.drop(columns=[top_asset])
        s_lo, _ = _book_cost(df2, lb0, 3, MAKER)
        print(f"   net (maker) WITHOUT top-held {top_asset.replace('USDT','')}: {_net(s_lo)}%  (vs {_net(s_mk)}% with)")
        # ROBUSTNESS: top-K x lookback (net, maker)
        print(f"   robustness net%(maker) by topK x lookback:")
        print(f"      {'':6}" + "".join(f"{'lb'+str(l):>8}" for l in [3, 7, 14]))
        for k in [1, 2, 3, 5]:
            row = f"      top{k:<3}"
            for l in [3, 7, 14]:
                s, _ = _book_cost(df, l, k, MAKER); row += f"{_net(s):>8}"
            print(row)
        print()
    print("[verdict] read above: if TAKER net << free net AND leave-one-out collapses it -> cost+concentration artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
