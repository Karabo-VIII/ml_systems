"""src/strat/mover_lab.py -- shared backtest LAB for the multi-strategy capture exploration.

USER 2026-06-19: lookback (14d signal) is DECOUPLED from holding -- exits are a free variable (bag profit,
carry winners, cut losers; the 3-day mark is a CHECKPOINT, not a forced flush). Explore a VARIETY of ways to
win (slow trend, weekly momentum, DAILY MOVERS, mean-reversion bounce, vol-breakout, asymmetric exits,
greedy concentration) and map where each wins / loses / teaches.

Every strategy = a function that builds a WEIGHT MATRIX W (dates x assets, long-only, row-sum<=1, no leverage).
`evaluate(W)` measures ALL strategies identically:
  - compound per year + full-cycle (2020-2022), maxDD
  - CHECKPOINT GREEN-RATE: fraction of non-overlapping H-day blocks (anchored to the 1st of each month) whose
    book return is > 0  ==  "profit within the window" (default H=3); reported per year too
  - avg gross exposure, avg daily turnover (cost already deducted)
Causal: positions are LAGGED 1 bar; taker cost on |dpos|. fixed-EW book over the weighted assets.

RWYB: python -m strat.mover_lab --selftest      (reproduces gated-beta + momentum reference numbers)
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.ma_strat_builder as msb
COST = msb.TAKER_RT


def _rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def load(start="2020-01-01", end="2023-01-01", syms=None):
    """Return a dict of aligned causal DataFrames + indicators (all past-only)."""
    A = msb._load_all("1d", start, end)
    if syms: A = [a for a in A if a["sym"] in syms]
    def df(k): return pd.DataFrame({a["sym"]: pd.Series(a[k], index=pd.to_datetime(a["ms"], unit="ms")) for a in A}).sort_index()
    C, O, H, L = df("c"), df("o"), df("h"), df("l")
    R = C.pct_change()
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()]).groupby(level=0).max()
    ind = {
        "C": C, "O": O, "H": H, "L": L, "R": R,
        "sma200": C.rolling(200, min_periods=200).mean(),
        "sma50": C.rolling(50, min_periods=50).mean(),
        "mom14": C / C.shift(14) - 1, "mom7": C / C.shift(7) - 1, "mom30": C / C.shift(30) - 1,
        "rsi14": C.apply(_rsi), "ret1": R,
        "hh14": C.rolling(14, min_periods=14).max(), "ll14": C.rolling(14, min_periods=14).min(),
        "vol20": R.rolling(20, min_periods=10).std() * np.sqrt(365),
        "atr14": tr.rolling(14, min_periods=14).mean(),
    }
    ind["gate"] = (C > ind["sma200"]).fillna(False)   # strict-ish (NaN warmup -> False)
    return ind


def _month_blocks(index, H):
    posix = np.arange(len(index)); per = index.to_period("M"); bl = []
    for p in pd.unique(per):
        mp = posix[per == p]
        for j in range(0, len(mp) - H + 1, H):
            bl.append(mp[j:j + H])
    return bl


def evaluate(W, ind, H=3, label=""):
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
    x = bret.to_numpy()
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    def comp(s, e):
        mask = np.asarray((bret.index >= s) & (bret.index < e)); xs = x[mask]
        return round((np.prod(1 + xs) - 1) * 100, 1) if mask.sum() > 2 else None
    # checkpoint green-rate (H-day blocks anchored to month-1st)
    blocks = _month_blocks(bret.index, H)
    gr = []
    for blk in blocks:
        r = np.prod(1 + x[blk]) - 1
        gr.append((bret.index[blk[0]], r))
    gdf = pd.DataFrame(gr, columns=["d", "r"]).set_index("d")
    def green(s, e):
        m = np.asarray((gdf.index >= s) & (gdf.index < e)); xs = gdf["r"].to_numpy()[m]
        return round(100 * float(np.mean(xs > 0)), 0) if len(xs) >= 3 else None
    nz = gdf["r"][gdf["r"] != 0.0]
    return {
        "label": label,
        "comp_2020": comp("2020-01-01", "2021-01-01"), "comp_2021": comp("2021-01-01", "2022-01-01"),
        "comp_2022": comp("2022-01-01", "2023-01-01"), "comp_full": round((eq[-1] - 1) * 100, 1),
        "maxDD": round(float(((eq - pk) / pk).min() * 100), 1),
        "green_all": round(100 * float(np.mean(nz.to_numpy() > 0)), 0) if len(nz) else 0.0,
        "green_2021": green("2021-01-01", "2022-01-01"), "green_2022": green("2022-01-01", "2023-01-01"),
        "avg_expo": round(float(pos.sum(axis=1).mean()), 2), "avg_turnover": round(float(turn.mean()), 3),
        "n_blocks_inmkt": int(len(nz)),
    }


# ----- reference strategies (for selftest + agents to build on) -----
def topk_weight(score, ind, K, gate=True, rebal=3):
    """Rebalance every `rebal` days: hold top-K by `score` among gated assets, EW; carry between rebals."""
    C = ind["C"]; g = ind["gate"] if gate else pd.DataFrame(True, index=C.index, columns=C.columns)
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns); last = -999
    for i, d in enumerate(C.index):
        if i - last >= rebal:
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            if elig:
                pick = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
                W.loc[d, :] = 0.0
                for s in pick: W.loc[d, s] = 1.0 / len(pick)
            last = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]
    return W


def selftest():
    ind = load()
    print("[selftest] mover_lab -- reference strategies")
    beta = (ind["gate"].astype(float)).div(ind["gate"].sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    print("  gated-beta:", {k: evaluate(beta, ind, label="beta")[k] for k in ("comp_2021", "comp_full", "maxDD", "green_2021")})
    mom7 = topk_weight(ind["mom14"], ind, K=3, rebal=7)
    print("  mom14 top3 rebal7:", {k: evaluate(mom7, ind, label="mom7")[k] for k in ("comp_2021", "comp_full", "maxDD", "green_2021")})
    print("[selftest] PASSED")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else 0)
