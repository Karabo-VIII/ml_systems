"""src/strat/referee_controls.py -- adversarial controls on the leaderboard.

(A) Same-exposure shuffle control on the regime-router: does the router's edge survive a
    control that holds its EXPOSURE PATH constant but SCRAMBLES which assets it picks?
    If a random pick at the same daily exposure does as well, the 'router skill' is just
    de-risked beta (cash the down-trends), not timing/selection.
(B) BH baseline robustness at large N (5000 slices) + across two OOS windows
    (2020-10 = gb_meta's window; 2022-01 = router's window) to show the baseline itself
    is a moving target -> WHY the original 6 reports disagree on '55%'.
(C) Router pos-rate at large N, and a 'never-negative' diagnostic: how often is the router
    actually flat/cash in down weeks.

No emoji (cp1252).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.referee_harness as rh
import strat.adaptive_meta_engine as ame


def shuffle_control(Wr: pd.DataFrame, ind: dict, n_shuffles: int, seed: int) -> pd.DataFrame:
    """Build a same-EXPOSURE-path shuffled book: each day keep the router's TOTAL exposure
    and number of held names, but pick which assets RANDOMLY from the assets with valid price.
    Returns a DataFrame of (shuffle book daily returns).
    """
    C = ind["C"]
    rng = np.random.default_rng(seed)
    present = C.notna()
    cols = list(C.columns)
    daily_expo = Wr.sum(axis=1)              # total exposure per day
    n_names = (Wr > 0).sum(axis=1)           # number of names held
    books = np.zeros((n_shuffles, len(C.index)))
    R = ind["R"].reindex(index=C.index, columns=C.columns).fillna(0.0).values
    col_pos = {c: j for j, c in enumerate(cols)}
    for sh in range(n_shuffles):
        Wsh = np.zeros((len(C.index), len(cols)))
        for i, d in enumerate(C.index):
            e = daily_expo.iloc[i]; k = int(n_names.iloc[i])
            if e <= 0 or k <= 0:
                continue
            avail = [c for c in cols if present.loc[d, c]]
            if not avail:
                continue
            k = min(k, len(avail))
            pick = rng.choice(len(avail), size=k, replace=False)
            w = e / k
            for pj in pick:
                Wsh[i, col_pos[avail[pj]]] = w
        # book return with 1-bar lag + cost
        pos = np.vstack([np.zeros((1, len(cols))), Wsh[:-1]])
        turn = np.abs(np.vstack([pos[:1], np.diff(pos, axis=0)])).sum(axis=1)
        bret = (pos * R).sum(axis=1) - turn * (rh.COST / 2.0)
        books[sh] = bret
    return pd.DataFrame(books.T, index=C.index)


def slice_posrate(bret: pd.Series, oos_start, oos_end, n, seed):
    rng = np.random.default_rng(seed)
    idx = bret.index
    m = (idx >= pd.Timestamp(oos_start)) & (idx < pd.Timestamp(oos_end))
    oos = idx[m]; ms = len(oos) - 7
    r = []
    for _ in range(n):
        si = rng.integers(0, ms)
        sl = oos[si:si + 7]
        r.append(float((1 + bret.loc[sl]).prod() - 1))
    r = np.array(r)
    return float((r > 0).mean()) * 100, float(r.mean()) * 100


def main():
    OOS_START, OOS_END, N = "2022-01-01", "2026-06-01", 5000
    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]
    bh_b = rh.book_daily_returns(rh.bh_ew_weights(ind), ind)

    print("=" * 76)
    print("(B) BH baseline robustness -- WHY the 6 reports disagree on '55%'")
    print("=" * 76)
    for (s, e, tag) in [("2020-10-17", "2026-06-01", "gb_meta window (2020-10+)"),
                        ("2020-01-06", "2023-01-01", "smartweight 2020-2023 bull"),
                        ("2022-01-01", "2026-06-01", "router OOS 2022+")]:
        prs = [slice_posrate(bh_b, s, e, N, sd)[0] for sd in [11, 23, 42]]
        mns = [slice_posrate(bh_b, s, e, N, sd)[1] for sd in [11, 23, 42]]
        print(f"  BH {tag:<32}: pos_rate={round(np.mean(prs),1)}% mean={round(np.mean(mns),2)}%")

    print("\n" + "=" * 76)
    print("(C) Router @ N=5000 (3 seeds)")
    print("=" * 76)
    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    Wr = ame.build_weight_matrix(ind, vthr)
    rb = rh.book_daily_returns(Wr, ind)
    rpr = [slice_posrate(rb, OOS_START, OOS_END, N, sd)[0] for sd in [11, 23, 42]]
    rmn = [slice_posrate(rb, OOS_START, OOS_END, N, sd)[1] for sd in [11, 23, 42]]
    bpr = [slice_posrate(bh_b, OOS_START, OOS_END, N, sd)[0] for sd in [11, 23, 42]]
    bmn = [slice_posrate(bh_b, OOS_START, OOS_END, N, sd)[1] for sd in [11, 23, 42]]
    print(f"  router : pos_rate={round(np.mean(rpr),1)}% (seeds {[round(x,1) for x in rpr]}) mean={round(np.mean(rmn),2)}%")
    print(f"  BH     : pos_rate={round(np.mean(bpr),1)}% (seeds {[round(x,1) for x in bpr]}) mean={round(np.mean(bmn),2)}%")

    print("\n" + "=" * 76)
    print("(A) SAME-EXPOSURE SHUFFLE CONTROL on the router (20 shuffles)")
    print("    null: keep router's daily exposure+#names, pick assets RANDOMLY.")
    print("    If shuffle ~ router -> 'router skill' is exposure timing (de-risked beta), NOT selection.")
    print("=" * 76)
    sh_books = shuffle_control(Wr, ind, n_shuffles=20, seed=7)
    sh_pr = []
    sh_mn = []
    for sh in sh_books.columns:
        pr, mn = slice_posrate(sh_books[sh], OOS_START, OOS_END, N, 42)
        sh_pr.append(pr); sh_mn.append(mn)
    router_pr_42, router_mn_42 = slice_posrate(rb, OOS_START, OOS_END, N, 42)
    sh_pr = np.array(sh_pr); sh_mn = np.array(sh_mn)
    # one-sided p: fraction of shuffles whose pos_rate >= router
    p_pr = float((sh_pr >= router_pr_42).mean())
    p_mn = float((sh_mn >= router_mn_42).mean())
    print(f"  router (seed42)     : pos_rate={round(router_pr_42,1)}%  mean={round(router_mn_42,2)}%")
    print(f"  shuffle null (n=20) : pos_rate mean={round(float(sh_pr.mean()),1)}% [p05 {round(float(np.percentile(sh_pr,5)),1)} p95 {round(float(np.percentile(sh_pr,95)),1)}]")
    print(f"                        mean_ret mean={round(float(sh_mn.mean()),2)}% [p05 {round(float(np.percentile(sh_mn,5)),2)} p95 {round(float(np.percentile(sh_mn,95)),2)}]")
    print(f"  one-sided p(shuffle>=router): pos_rate p={p_pr:.3f}  mean p={p_mn:.3f}")
    print(f"  VERDICT: router selection {'SURVIVES' if (p_pr<0.05 or p_mn<0.05) else 'does NOT survive'} the same-exposure shuffle.")

    out = {
        "bh_window_sensitivity": "see stdout",
        "router_N5000": {"pos_rate": round(float(np.mean(rpr)), 1), "mean": round(float(np.mean(rmn)), 2)},
        "bh_N5000": {"pos_rate": round(float(np.mean(bpr)), 1), "mean": round(float(np.mean(bmn)), 2)},
        "shuffle_control": {"router_pr": round(router_pr_42, 1), "router_mn": round(router_mn_42, 2),
                            "shuffle_pr_mean": round(float(sh_pr.mean()), 1),
                            "shuffle_mn_mean": round(float(sh_mn.mean()), 2),
                            "p_posrate": p_pr, "p_mean": p_mn},
    }
    (ROOT.parent / "runs" / "strat" / "referee_controls_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved referee_controls_results.json")


if __name__ == "__main__":
    main()
