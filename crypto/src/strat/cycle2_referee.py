"""src/strat/cycle2_referee.py -- INDEPENDENT adversarial re-derivation of Cycle-2 OOS claims.

Does NOT import any lane agent's script. Builds W matrices from mover_lab primitives only,
computes per-year compound (incl 2023/2024/2025 which evaluate() does not natively report),
and runs the make-or-break OOS shuffle null with a FIXED seed so the p-value is reproducible.

RWYB. No emoji (cp1252).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.mover_lab as ml

COST = ml.COST  # taker round-trip


# ---------- independent equity / per-year engine (re-derived, not ml.evaluate) ----------
def book_returns(W, ind):
    """Daily book return stream, lag-1, taker cost on |dpos|. Mirrors ml.evaluate mechanics."""
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
    return bret, pos, turn


def year_comp(bret, y):
    s = pd.Timestamp(f"{y}-01-01"); e = pd.Timestamp(f"{y+1}-01-01")
    m = (bret.index >= s) & (bret.index < e)
    x = bret[m].to_numpy()
    return (np.prod(1 + x) - 1) * 100 if m.sum() > 2 else None


def window_comp(bret, s, e):
    m = (bret.index >= pd.Timestamp(s)) & (bret.index < pd.Timestamp(e))
    x = bret[m].to_numpy()
    return (np.prod(1 + x) - 1) * 100 if m.sum() > 2 else None


def maxdd(bret, s=None, e=None):
    b = bret
    if s is not None:
        b = bret[(bret.index >= pd.Timestamp(s)) & (bret.index < pd.Timestamp(e))]
    x = b.to_numpy()
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100) if len(x) else 0.0


def summary(W, ind, name):
    bret, pos, turn = book_returns(W, ind)
    eq = np.cumprod(1 + bret.to_numpy())
    row = {"name": name}
    for y in range(2020, 2026):
        row[str(y)] = year_comp(bret, y)
    row["FULL"] = (eq[-1] - 1) * 100
    row["maxDD"] = maxdd(bret)
    row["expo"] = float(pos.sum(axis=1).mean())
    row["turn"] = float(turn.mean())
    return row, bret


# ---------- strategy builders (re-derived from primitives) ----------
def gated_beta(ind):
    g = ind["gate"].astype(float)
    return g.div(g.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def buy_hold(ind):
    C = ind["C"]
    return pd.DataFrame(1.0 / C.shape[1], index=C.index, columns=C.columns)


def breakout_score(ind):
    # distance above 14-day high proxy: C / hh14 (>1 means new high). Higher = stronger breakout.
    return ind["C"] / ind["hh14"]


def mr_score(ind):
    # mean-reversion: low rsi14 = more oversold = higher score
    return -ind["rsi14"]


# ---------- the make-or-break: OOS shuffle null ----------
def random_gated_k(ind, K, rebal, rng):
    """Random-K pick among gated assets, EW, carried between rebals. Same exposure profile as topk."""
    C = ind["C"]; g = ind["gate"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns); last = -999
    cols = list(C.columns)
    for i, d in enumerate(C.index):
        if i - last >= rebal:
            elig = [s for s in cols if bool(g.loc[d, s])]
            if elig:
                k = min(K, len(elig))
                pick = list(rng.choice(elig, size=k, replace=False))
                W.loc[d, :] = 0.0
                for s in pick:
                    W.loc[d, s] = 1.0 / k
            last = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]
    return W


def shuffle_null(score, ind, K, rebal, win_s, win_e, n_seeds=200, base_seed=12345):
    """True topk vs n_seeds random-gated-K, compound over [win_s, win_e). One-sided p = P(null >= true)."""
    Wt = ml.topk_weight(score, ind, K=K, gate=True, rebal=rebal)
    bt, _, _ = book_returns(Wt, ind)
    true_c = window_comp(bt, win_s, win_e)
    nulls = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        Wn = random_gated_k(ind, K, rebal, rng)
        bn, _, _ = book_returns(Wn, ind)
        nulls.append(window_comp(bn, win_s, win_e))
    nulls = np.array([n for n in nulls if n is not None], float)
    p = float((np.sum(nulls >= true_c) + 1) / (len(nulls) + 1))  # +1 smoothing
    return {
        "true": true_c, "null_med": float(np.median(nulls)),
        "null_p05": float(np.percentile(nulls, 5)), "null_p95": float(np.percentile(nulls, 95)),
        "p_value": p, "n_seeds": len(nulls),
    }


def jackknife_drop_asset(score, ind, K, rebal, win_s, win_e):
    """Leave-one-asset-out: how much of the OOS edge depends on a single name?"""
    cols = list(ind["C"].columns)
    out = {}
    Wfull = ml.topk_weight(score, ind, K=K, gate=True, rebal=rebal)
    bf, _, _ = book_returns(Wfull, ind)
    out["ALL"] = window_comp(bf, win_s, win_e)
    for drop in cols:
        keep = [c for c in cols if c != drop]
        sub = {k: (v[keep] if hasattr(v, "columns") else v) for k, v in ind.items()
               if k in ("C", "O", "H", "L", "R", "gate", "mom14", "rsi14", "hh14", "ll14", "sma200", "sma50",
                        "mom7", "mom30", "ret1", "vol20", "atr14")}
        sc = score[keep]
        W = ml.topk_weight(sc, sub, K=K, gate=True, rebal=rebal)
        b, _, _ = book_returns(W, sub)
        out[f"drop_{drop}"] = window_comp(b, win_s, win_e)
    return out


def main():
    print("[referee] loading u10 2020-2026 ...")
    ind = ml.load("2020-01-01", "2026-06-01")

    strategies = {
        "gated_beta": gated_beta(ind),
        "buy_hold_EW": buy_hold(ind),
        "mom14_K5_r3": ml.topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=3),
        "mom14_K5_r14": ml.topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=14),
        "breakout_K3_r5": ml.topk_weight(breakout_score(ind), ind, K=3, gate=True, rebal=5),
        "MR_rsi_K3_r5": ml.topk_weight(mr_score(ind), ind, K=3, gate=True, rebal=5),
    }

    rows = []
    brets = {}
    for nm, W in strategies.items():
        row, bret = summary(W, ind, nm)
        rows.append(row); brets[nm] = bret

    print("\n=== PER-YEAR COMPOUND (%) [independently re-derived] ===")
    hdr = ["name"] + [str(y) for y in range(2020, 2026)] + ["FULL", "maxDD", "expo", "turn"]
    print(" | ".join(f"{h:>12}" for h in hdr))
    for r in rows:
        def f(v):
            return "----" if v is None else f"{v:,.1f}"
        print(" | ".join([f"{r['name']:>12}"] + [f"{f(r[str(y)]):>12}" for y in range(2020, 2026)]
                         + [f"{r['FULL']:>12,.0f}", f"{r['maxDD']:>12.1f}", f"{r['expo']:>12.2f}", f"{r['turn']:>12.3f}"]))

    # ---- MAKE-OR-BREAK: OOS shuffle null, per-year + pooled, mom14-K5 r3 (lane oos_forward used r14;
    #      I test BOTH the cycle-1 frozen config r14 AND the r3 the gate lane used, to be fair) ----
    print("\n=== OOS SHUFFLE NULL: mom14-K5 vs random-gated-5 (N=200 fixed-seed) ===")
    oos = ("2023-01-01", "2026-06-01")
    for rebal in (3, 14):
        sc = ind["mom14"]
        res = shuffle_null(sc, ind, K=5, rebal=rebal, win_s=oos[0], win_e=oos[1], n_seeds=200)
        print(f"\n-- mom14 K5 rebal={rebal}, OOS 2023..2026 --")
        print(f"   true={res['true']:,.1f}%  null_med={res['null_med']:,.1f}%  "
              f"p05={res['null_p05']:,.1f}  p95={res['null_p95']:,.1f}  p={res['p_value']:.3f}  (n={res['n_seeds']})")
        # per-year
        for y in range(2023, 2026):
            r = shuffle_null(sc, ind, K=5, rebal=rebal, win_s=f"{y}-01-01", win_e=f"{y+1}-01-01", n_seeds=200)
            print(f"   {y}: true={r['true']:>8,.1f}%  null_med={r['null_med']:>8,.1f}%  p={r['p_value']:.3f}")

    # ---- breakout OOS shuffle null (vs random-gated-3) ----
    print("\n=== OOS SHUFFLE NULL: breakout-K3 vs random-gated-3 (N=200) ===")
    rb = shuffle_null(breakout_score(ind), ind, K=3, rebal=5, win_s=oos[0], win_e=oos[1], n_seeds=200)
    print(f"   OOS true={rb['true']:,.1f}%  null_med={rb['null_med']:,.1f}%  "
          f"p05={rb['null_p05']:,.1f}  p95={rb['null_p95']:,.1f}  p={rb['p_value']:.3f}")
    for y in range(2023, 2026):
        r = shuffle_null(breakout_score(ind), ind, K=3, rebal=5, win_s=f"{y}-01-01", win_e=f"{y+1}-01-01", n_seeds=200)
        print(f"   {y}: true={r['true']:>8,.1f}%  null_med={r['null_med']:>8,.1f}%  p={r['p_value']:.3f}")

    # ---- jackknife drop-asset on the OOS window (single-name dependence) ----
    print("\n=== JACKKNIFE drop-one-asset: mom14-K5 r3 OOS 2023..2026 compound (%) ===")
    jk = jackknife_drop_asset(ind["mom14"], ind, K=5, rebal=3, win_s=oos[0], win_e=oos[1])
    base = jk["ALL"]
    for k, v in sorted(jk.items(), key=lambda kv: (kv[0] != "ALL", kv[1] if kv[1] is not None else 0)):
        delta = "" if k == "ALL" else f"  (delta {v-base:+,.1f})"
        print(f"   {k:>16}: {v:>10,.1f}%{delta}")


if __name__ == "__main__":
    main()
