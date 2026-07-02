"""QUANT probe: is MOM8_ER's edge momentum-RANK skill, or just be-long-during-gate-on-weeks (exposure/regime)?

Reuses the real book (strat_prod_deploy.build_all) for roo/idx/cols/net_of/gate/vol, and reconstructs the EXACT
mom8er selection internals (mom14, close finiteness, gate) so the rank-slice variants differ ONLY in which slice of
the momentum-sorted eligible set is chosen. Same gate, same eligibility, same inv-vol cap33 sizing, same causal cost.

Nulls:
 (A) TOP-8  = the real book (control).
 (B) MID-8  = the 8 names straddling the median momentum rank in the eligible set.
 (C) BOT-8  = the 8 LOWEST-momentum eligible names (anti-momentum).
 (D) eligible-set random null: at each deploy week draw 8 names uniformly from the SAME gate-eligible set, inv-vol
     size them the same way, 500 draws. Percentile of TOP-8 compound vs the 500-draw distribution.

Decisive: TOP-8 >> BOT-8 (rank carries signal) AND TOP-8 clears the eligible-set random null > 95th pctile.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[3] / "src"; sys.path.insert(0, str(ROOT))
import strat_prod_deploy as P
from strat.canonical_exits import COST_PROD

OOS = pd.Timestamp("2024-05-15")
CAP = P.CAP; SL = P.SL; BPD = P.BPD


def build_internals():
    """Reconstruct the EXACT panel mom8er() sees: C (closes), mom14, gate, vol, plus roo/idx/cols/net_of."""
    f = P.build_all()
    idx = f["idx"]; cols = f["cols"]; T = f["T"]; A = f["A"]
    roo = f["roo"]; vol = f["vol"]; gbin = f["gbin"]; er = f["er"]; net_of = f["net_of"]
    gate = gbin * (er > 0.35).astype(int)
    # rebuild closes panel identically to build_all
    closes = {}
    for s in P._u100_symbols():
        r = P._load_asset(s, "1d", P.WALL)
        if r is None: continue
        d, o, h, l, c = r
        closes[s] = pd.Series(np.asarray(c, float), index=d)
    cdf = pd.DataFrame(closes).sort_index().reindex(idx).reindex(columns=cols)
    C = cdf.to_numpy()
    mom14 = cdf.pct_change(14 * BPD).to_numpy()
    starts = [i for i in range(0, T - SL + 1, SL)]
    return dict(idx=idx, cols=cols, T=T, A=A, roo=roo, vol=vol, gate=gate,
                C=C, mom14=mom14, starts=starts, net_of=net_of)


def size_inv_vol(vol_i, sel):
    """inv-vol weights over the selected indices, capped at CAP; equal-weight fallback (mirrors mom8er exactly)."""
    v = vol_i[sel]; w = np.where(np.isfinite(v) & (v > 0), 1.0 / v, 0.0)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(sel)) / len(sel)
    return np.minimum(w, CAP)


def book_from_selector(G, selector):
    """Build W[T,A] by applying `selector(eligible_sorted_desc, i) -> chosen idx array` at each gated deploy week."""
    T, A = G["T"], G["A"]; W = np.zeros((T, A))
    mom14 = G["mom14"]; C = G["C"]; gate = G["gate"]; vol = G["vol"]
    for i in G["starts"]:
        if gate[i] <= 0: continue
        s = mom14[i]; e = np.where(np.isfinite(s) & np.isfinite(C[i]))[0]
        if len(e) == 0: continue
        order = e[np.argsort(-s[e])]          # eligible names, momentum DESCENDING (best first)
        sel = selector(order, i)
        if sel is None or len(sel) == 0: continue
        w = size_inv_vol(vol[i], sel)
        W[i:i + SL, sel] = w
    return W


def compound_oos(G, W):
    net = G["net_of"](W)
    om = G["idx"] >= OOS
    return float(np.prod(1 + net[om]) - 1.0) * 100.0


def sel_top(order, i):
    return order[:min(8, len(order))]


def sel_bottom(order, i):
    k = min(8, len(order)); return order[-k:]         # lowest momentum


def sel_middle(order, i):
    n = len(order); k = min(8, n)
    if n <= 8: return order[:k]
    lo = (n - k) // 2
    return order[lo:lo + k]                            # straddle the median rank


def main():
    G = build_internals()
    # deploy-week diagnostics
    ndeploy = sum(1 for i in G["starts"] if G["gate"][i] > 0)
    elig_sizes = [len(np.where(np.isfinite(G["mom14"][i]) & np.isfinite(G["C"][i]))[0])
                  for i in G["starts"] if G["gate"][i] > 0]
    oos_deploy = sum(1 for i in G["starts"] if G["gate"][i] > 0 and G["idx"][i] >= OOS)

    Wt = book_from_selector(G, sel_top)
    Wm = book_from_selector(G, sel_middle)
    Wb = book_from_selector(G, sel_bottom)
    c_top = compound_oos(G, Wt); c_mid = compound_oos(G, Wm); c_bot = compound_oos(G, Wb)

    # sanity: our reconstructed TOP-8 must match the real book's MOM8_ER OOS compound
    f = P.build_all()
    real = compound_oos({**G, "net_of": f["net_of"]}, f["CAND"]["MOM8_ER"])

    # (D) eligible-set random null: 500 draws of 8-from-eligible per deploy week
    ND = 500; rng = np.random.default_rng(12345); draws = np.empty(ND)
    for d in range(ND):
        def sel_rand(order, i, _rng=rng):
            k = min(8, len(order)); return _rng.choice(order, size=k, replace=False)
        Wd = book_from_selector(G, sel_rand)
        draws[d] = compound_oos(G, Wd)
    pct_top = float((draws < c_top).mean() * 100.0)
    pct_bot = float((draws < c_bot).mean() * 100.0)

    out = dict(
        n_deploy_weeks=ndeploy, oos_deploy_weeks=oos_deploy,
        elig_min=int(min(elig_sizes)), elig_median=float(np.median(elig_sizes)), elig_max=int(max(elig_sizes)),
        real_MOM8_ER_oos_pct=round(real, 2),
        reconstructed_top8_oos_pct=round(c_top, 2),
        middle8_oos_pct=round(c_mid, 2),
        bottom8_oos_pct=round(c_bot, 2),
        top_minus_bottom_pp=round(c_top - c_bot, 2),
        top_minus_middle_pp=round(c_top - c_mid, 2),
        eligible_random_null_n=ND,
        eligible_null_mean_pct=round(float(draws.mean()), 2),
        eligible_null_p05_pct=round(float(np.percentile(draws, 5)), 2),
        eligible_null_p50_pct=round(float(np.percentile(draws, 50)), 2),
        eligible_null_p95_pct=round(float(np.percentile(draws, 95)), 2),
        eligible_null_max_pct=round(float(draws.max()), 2),
        top8_percentile_vs_eligible_null=round(pct_top, 1),
        bottom8_percentile_vs_eligible_null=round(pct_bot, 1),
    )
    print(json.dumps(out, indent=2))
    with open(HERE / "quant_mom_rank_null.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
