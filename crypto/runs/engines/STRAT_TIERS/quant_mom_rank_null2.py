"""QUANT probe 2: robustness of the 92.4-pctile + the top>mid>bot gradient.
 - seed-stability: recompute the eligible-set null percentile of TOP-8 across 5 independent seeds (2000 draws each).
 - leave-one-deploy-week-out jackknife on the OOS window: does TOP-8 still beat BOT-8, and does the pctile hold,
   when we drop each single OOS deploy week? (isolates lumpiness / single-week dependence).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[3] / "src"; sys.path.insert(0, str(ROOT))
import quant_mom_rank_null as Q
import strat_prod_deploy as P

OOS = Q.OOS


def per_deploy_net(G, selector):
    """Return per-DEPLOY-WEEK net compound (product over the 7-bar slice) for OOS deploy weeks, in order."""
    idx = G["idx"]; net_full = G["net_of"](Q.book_from_selector(G, selector))
    rows = []
    for i in G["starts"]:
        if G["gate"][i] <= 0: continue
        if idx[i] < OOS: continue
        sl = net_full[i + 1:i + 1 + P.SL]   # slice returns realized (weights lag 1 bar; net indexed by realize bar)
        rows.append(float(np.prod(1 + sl) - 1.0))
    return np.array(rows)


def main():
    G = Q.build_internals()

    # per-deploy-week compound for top/mid/bot (used for jackknife)
    top = per_deploy_net(G, Q.sel_top)
    mid = per_deploy_net(G, Q.sel_middle)
    bot = per_deploy_net(G, Q.sel_bottom)
    # full-window compound recomputed from per-week (chained) as a cross-check
    def chain(x): return float(np.prod(1 + x) - 1.0) * 100.0
    top_c, mid_c, bot_c = chain(top), chain(mid), chain(bot)

    # jackknife: drop each single OOS deploy week, recompute chained compound
    jk = []
    for k in range(len(top)):
        m = np.ones(len(top), bool); m[k] = False
        jk.append(dict(drop_week=k,
                       top=round(chain(top[m]), 1), mid=round(chain(mid[m]), 1),
                       bot=round(chain(bot[m]), 1),
                       top_gt_bot=bool(chain(top[m]) > chain(bot[m]))))
    # worst-case drop for TOP and the resulting top-vs-bot
    jk_top_vals = [j["top"] for j in jk]
    worst_top_drop = int(np.argmin(jk_top_vals))
    top_still_beats_bot_all = all(j["top_gt_bot"] for j in jk)

    # seed-stability of the eligible-set null percentile of TOP-8 (full-window compound)
    c_top_full = Q.compound_oos(G, Q.book_from_selector(G, Q.sel_top))
    pcts = []
    for seed in [1, 7, 42, 2024, 99999]:
        rng = np.random.default_rng(seed); ND = 2000; draws = np.empty(ND)
        for d in range(ND):
            def sel_rand(order, i, _rng=rng):
                kk = min(8, len(order)); return _rng.choice(order, size=kk, replace=False)
            draws[d] = Q.compound_oos(G, Q.book_from_selector(G, sel_rand))
        pcts.append(round(float((draws < c_top_full).mean() * 100.0), 1))

    out = dict(
        top_chain_pct=round(top_c, 1), mid_chain_pct=round(mid_c, 1), bot_chain_pct=round(bot_c, 1),
        per_week_top=[round(x * 100, 1) for x in top],
        per_week_bot=[round(x * 100, 1) for x in bot],
        n_oos_weeks=len(top),
        n_positive_top=int((top > 0).sum()), n_positive_bot=int((bot > 0).sum()),
        jackknife_top_range=[min(jk_top_vals), max(jk_top_vals)],
        worst_case_drop_week=worst_top_drop,
        top_after_worst_drop=jk_top_vals[worst_top_drop],
        top_beats_bot_every_jackknife=top_still_beats_bot_all,
        eligible_null_pctile_5seeds=pcts,
        eligible_null_pctile_mean=round(float(np.mean(pcts)), 1),
    )
    print(json.dumps(out, indent=2))
    with open(HERE / "quant_mom_rank_null2.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
