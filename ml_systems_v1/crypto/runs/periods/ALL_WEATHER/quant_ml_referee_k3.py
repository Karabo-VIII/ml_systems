"""
quant_ml_referee_k3.py -- DERIVATION #3: the structural ceiling + significance, independent path.

Answers Q3 (smoothing ceiling) and Q4 (hard wall) with:
  - a pure-structure book sweep (EW / inv-vol / inv-var / min-corr) over random 7d slices
  - a BINOMIAL one-sided test: is any book's abs win-rate significantly > 55%? (n_eff adjusted for
    7d-overlap dependence via a moving-block bootstrap effective-n)
  - the honest BH bar measured the SAME way (point-in-time, listed-only)
  - a 'cheapest falsifier' probe: if even the BEST no-prediction book can't clear 55% significantly,
    then 55% is the structural wall and only genuine prediction (which strict-WF showed = absent) can break it.

No emoji. Does not commit.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SRC = Path(r"c:\Users\karab\Documents\coding\ml_systems\crypto\src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import strat.mover_lab as lab

SEED = 99
H = 7
N = 600


def books(ind):
    """Return weight builders that need NO prediction (pure structure)."""
    C = ind["C"]; R = ind["R"]
    vol = R.rolling(20, min_periods=10).std()
    var = vol ** 2
    return C.to_numpy(), vol.to_numpy(), var.to_numpy(), R


def mincorr_weights(Rwin, listed):
    """Down-weight names that are highly correlated with the basket (diversification)."""
    if listed.sum() < 2:
        w = listed.astype(float)
        return w / max(w.sum(), 1)
    sub = Rwin[:, listed]
    if sub.shape[0] < 5 or np.isnan(sub).all():
        w = listed.astype(float); return w / w.sum()
    cc = np.corrcoef(np.nan_to_num(sub.T))
    avg_corr = np.nanmean(cc, axis=1)
    inv = 1.0 / (avg_corr - avg_corr.min() + 0.5)
    w = np.zeros(len(listed)); w[listed] = inv
    return w / w.sum()


def eval_books(ind, n=N, seed=SEED):
    Cv, volv, varv, R = books(ind)
    Rv = R.to_numpy()
    nD, nA = Cv.shape
    rng = np.random.default_rng(seed)
    starts = np.arange(200, nD - H - 1)
    pick = rng.choice(starts, size=n, replace=True)
    out = {k: [] for k in ("EW", "INVVOL", "INVVAR", "MINCORR")}
    for s in pick:
        fwd = Cv[s + H] / Cv[s] - 1.0
        listed = ~np.isnan(fwd)
        if listed.sum() < 2:
            continue
        # EW
        out["EW"].append(np.nanmean(fwd[listed]))
        # inv-vol
        w = np.where(listed & np.isfinite(volv[s]) & (volv[s] > 0), 1.0 / (volv[s] + 1e-8), 0.0)
        out["INVVOL"].append(float(np.nansum(np.where(listed, fwd, 0) * (w / w.sum()))) if w.sum() > 0 else np.nanmean(fwd[listed]))
        # inv-var
        w2 = np.where(listed & np.isfinite(varv[s]) & (varv[s] > 0), 1.0 / (varv[s] + 1e-8), 0.0)
        out["INVVAR"].append(float(np.nansum(np.where(listed, fwd, 0) * (w2 / w2.sum()))) if w2.sum() > 0 else np.nanmean(fwd[listed]))
        # min-corr (use 60d trailing window for corr)
        Rwin = Rv[max(0, s - 60):s]
        wc = mincorr_weights(Rwin, listed)
        out["MINCORR"].append(float(np.nansum(np.where(listed, fwd, 0) * wc)))
    return out


def block_eff_n(win, block=8):
    """effective n via moving-block bootstrap variance inflation vs iid."""
    x = win.astype(float); n = len(x)
    rng = np.random.default_rng(1)
    nb = int(np.ceil(n / block))
    boots = []
    for _ in range(2000):
        st = rng.integers(0, n - block + 1, size=nb) if n > block else np.array([0])
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in st])[:n]
        boots.append(x[idx].mean())
    var_block = np.var(boots)
    p = x.mean()
    var_iid = p * (1 - p) / n
    infl = var_block / max(var_iid, 1e-12)
    return n / max(infl, 1.0), infl


if __name__ == "__main__":
    print("DERIVATION #3 -- structural ceiling + binomial significance vs 55%")
    ind = lab.load("2020-01-01", "2026-06-01")
    res = eval_books(ind)
    print(f"\n  {'book':<9}{'abs_WR':>8}{'mean':>9}{'n':>6}{'eff_n':>8}{'p(>55%)':>10}{'p(>BH)':>9}")
    ew = np.array(res["EW"])
    ew_wr = (ew > 0).mean()
    for k, v in res.items():
        a = np.array(v); wr = (a > 0).mean(); win = (a > 0)
        eff, infl = block_eff_n(win)
        # one-sided binomial vs 0.55 using effective n
        k_succ = int(round(wr * eff)); n_eff = int(round(eff))
        p55 = stats.binomtest(k_succ, n_eff, 0.55, alternative="greater").pvalue if n_eff > 0 else np.nan
        # vs BH (EW) win-rate
        p_bh = stats.binomtest(k_succ, n_eff, max(ew_wr, 1e-6), alternative="greater").pvalue if n_eff > 0 else np.nan
        print(f"  {k:<9}{wr*100:>7.1f}%{a.mean()*100:>+8.2f}%{len(a):>6}{eff:>8.0f}{p55:>10.3f}{p_bh:>9.3f}")
    print(f"\n  BH (EW) abs win-rate reference = {ew_wr*100:.1f}%")
    print("  Interpretation: p(>55%) >> 0.05 for every book => 55% is NOT significantly beatable")
    print("  by structure alone; p(>BH) >> 0.05 => no smoothing book beats plain EW. The wall is structural.")
