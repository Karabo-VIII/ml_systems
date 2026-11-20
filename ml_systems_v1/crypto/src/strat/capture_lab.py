"""src/strat/capture_lab.py -- MOVE-CATCH capture-rate harness (user's binding directive 2026-06-18 + referee lever #2).

Judges a TI by CAPTURE-RATE = realized / available move within the signal window -- NOT wealth-vs-buy-hold, and NOT the
same-exposure shuffle (the sub-daily referee proved that null is BROKEN: per-bar reselection churn-penalizes the control).
The move-CATCH thesis (TREND-FOLLOWING, prediction failed): a TI fires when a move is underway -> ENTER -> ride -> EXIT by
mechanism -> measure the fraction of the AVAILABLE up-move captured. The null is RANDOM ENTRY at matched frequency
(churn-IMMUNE: each entry is one independent per-signal trade held to a mechanism exit, ~1 RT -- no per-bar reselection).

Three distinct questions, all reported:
  (A) IDENTIFICATION  -- mean MFE | TI fires  vs | random entry   (does the TI find bigger available moves?)
  (B) TRADEABLE       -- mean realized net    | TI fires  vs | random   (does entering on it make money net of cost?)
  (C) CAPTURE-EFFY    -- mean realized / MFE  (given a move, what fraction did the mechanism keep?)

DEV-walled via fleet_lab.load_wide (<= 2024-05-15). Long-only spot, taker cost, causal triggers. No emoji (cp1252).
RWYB: python -m strat.capture_lab --selftest [--tf 4h]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.fleet_lab as fl

COST = fl.COST


# ---------- available move + mechanism exits ----------
def mfe_matrix(C, hold):
    """Max Favorable Excursion (available up-move, perfect-exit oracle) for every (di, asset): max(C[di+1..di+hold])/C[di]-1."""
    fwd_max = C.rolling(hold, min_periods=1).max().shift(-hold)     # at di -> max over (di+1 .. di+hold)
    return fwd_max / C - 1.0


def time_return_matrix(C, hold):
    """Hold-to-maturity (time-stop) net return for every (di, asset): C[di+hold]/C[di]-1 - cost."""
    return C.shift(-hold) / C - 1.0 - COST


def block_edge_p(rows_f, vals_f, rows_p, vals_p, n_bars, block_bars=21, n_boot=1500, seed=1):
    """HONEST date-block moving-block bootstrap of the edge (mean fired - mean pool). Resamples contiguous bar-blocks
    (respects the heavy 7d-hold/cross-asset overlap) -- unlike iid entry-resampling, which deflates SE ~2-3.6x (the
    referee's root-cause fix, 2026-06-20). rows_* are bar-row indices into C.index; block_bars = ~21 calendar days in bars."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    fd = defaultdict(list); pdd = defaultdict(list)
    for v, d in zip(vals_f, rows_f): fd[int(d)].append(v)
    for v, d in zip(vals_p, rows_p): pdd[int(d)].append(v)
    obs = float(np.mean(vals_f) - np.mean(vals_p))
    n_blocks = max(1, n_bars // block_bars)
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n_bars - block_bars + 1), size=n_blocks)
        fv = []; pv = []
        for s in starts:
            for d in range(int(s), int(s) + block_bars):
                if d in fd: fv.extend(fd[d])
                if d in pdd: pv.extend(pdd[d])
        if len(fv) >= 5 and len(pv) >= 5:
            boots.append(np.mean(fv) - np.mean(pv))
    boots = np.array(boots)
    return {"obs_edge_pp": round(100 * obs, 3),
            "block_p05_pp": round(100 * float(np.percentile(boots, 5)), 3) if len(boots) else None,
            "block_p_le0": round(float(np.mean(boots <= 0)), 4) if len(boots) else None,
            "n_eff_dates": int(len(np.unique(rows_f)))}


def exit_return(path, p0, trail=None, target=None):
    """Realized long net return on a price path (entry p0, path = C[di+1..di+hold]) with a MECHANISM exit.
    trail = trailing-stop frac (e.g. 0.05); target = profit-target frac. time-stop = hold to end. Causal (forward path)."""
    if not np.isfinite(p0) or p0 <= 0:
        return np.nan
    path = path[np.isfinite(path)]
    if len(path) == 0:
        return np.nan
    peak = p0
    for px in path:
        r = px / p0 - 1.0
        if target is not None and r >= target:
            return target - COST
        peak = max(peak, px)
        if trail is not None and px <= peak * (1.0 - trail):
            return float(px / p0 - 1.0) - COST
    return float(path[-1] / p0 - 1.0) - COST


# ---------- causal regime + TI move-onset trigger ----------
def regime_series(lab, tf="1d", w_days=50):
    """Causal bull/chop/bear per bar: BTC(or EW-proxy) vs its rolling mean x universe breadth. Returns Series of labels."""
    C = lab["C"]; bpd = fl.BARS_PER_DAY[tf]; W = max(10, w_days * bpd)
    btc = C["BTC"] if "BTC" in C.columns else C.mean(axis=1)
    trend = (btc / btc.rolling(W, min_periods=W // 2).mean() - 1.0)
    above = C.gt(C.rolling(W, min_periods=W // 2).mean())
    breadth = above.sum(axis=1) / above.notna().sum(axis=1).replace(0, np.nan)
    reg = pd.Series("chop", index=C.index)
    reg[(trend > 0) & (breadth > 0.5)] = "bull"
    reg[(trend < 0) & (breadth < 0.5)] = "bear"
    return reg


def fired_matrix(lab, ti, thr=None):
    """Boolean DataFrame: where TI signals a move is ONSET/underway (causal, per-asset, cross-sectional where needed).
    If the stored feature is already a boolean DataFrame (pre-computed gate), return it directly."""
    F = lab["F"]; X = F[ti]
    # pre-computed boolean gate (e.g. combined price-TI x exo conditioner) -- return as-is
    if hasattr(X, "dtype") and X.dtype == bool:
        return X.fillna(False)
    if hasattr(X, "dtypes") and (X.dtypes == bool).all():
        return X.fillna(False)
    if ti in ("mom7", "mom14", "mom30", "accel", "brk14"):
        return X > (0.0 if thr is None else thr)               # positive momentum / fresh breakout
    if ti == "rsi14":
        return X > (55 if thr is None else thr)                # momentum regime (not oversold-bounce)
    if ti == "rangepos":
        return X > (0.7 if thr is None else thr)               # near top of range
    if ti == "volexp":
        return X > (1.2 if thr is None else thr)               # vol expansion = move underway
    return X.gt(X.quantile(0.66, axis=1), axis=0)              # chimera/other: top cross-sectional tercile


# ---------- the capture-rate evaluation (TI vs random-ENTRY null) ----------
def evaluate_ti(lab, ti, tf="1d", hold=None, exit_kind="time", min_move=0.03, warm=40,
                n_null=300, seed=0, thr=None, by_regime=False, max_entries=12000, block=False):
    """Capture-rate of a TI vs a matched RANDOM-ENTRY null. exit_kind in {time, trail2, trail5, target8}.
    The 'time' exit is fully vectorized; mechanism exits loop per-entry (sub-sampled to max_entries for tractability).
    block=True adds an HONEST date-block bootstrap p (the iid p_vs_random deflates SE 2-3.6x on overlapping entries)."""
    C = lab["C"]; bpd = fl.BARS_PER_DAY[tf]
    hold = hold if hold is not None else 7 * bpd
    MFE = mfe_matrix(C, hold); TIME = time_return_matrix(C, hold)
    reg = regime_series(lab, tf) if by_regime else None
    fired = fired_matrix(lab, ti, thr)
    n = len(C.index)
    # valid (di, asset) entry universe: warm <= di <= n-hold-1, price + MFE finite
    valid = np.zeros((n, len(C.columns)), dtype=bool)
    valid[warm:n - hold - 1, :] = True
    MFEa = MFE.to_numpy(); TIMEa = TIME.to_numpy(); Ca = C.to_numpy()
    valid &= np.isfinite(Ca) & np.isfinite(MFEa)
    rng = np.random.default_rng(seed)

    def realized_path(di, jcol):
        p0 = Ca[di, jcol]; path = Ca[di + 1:di + hold + 1, jcol]
        if exit_kind == "trail2": return exit_return(path, p0, trail=0.02)
        if exit_kind == "trail5": return exit_return(path, p0, trail=0.05)
        if exit_kind == "target8": return exit_return(path, p0, target=0.08, trail=0.05)
        return exit_return(path, p0)

    fmat = fired.to_numpy() & valid & (MFEa > min_move)            # only windows with an available move to catch
    poolmat = valid & (MFEa > min_move)
    f_idx = np.array(np.where(fmat)).T                             # (n_fired, 2)
    p_idx = np.array(np.where(poolmat)).T
    if len(f_idx) < 30:
        return {"ti": ti, "tf": tf, "exit": exit_kind, "n_fired": int(len(f_idx)), "note": "insufficient signals"}
    # sub-sample for mechanism exits (the per-entry path loop is the only slow part); 'time' uses all entries.
    if exit_kind != "time":
        if len(f_idx) > max_entries: f_idx = f_idx[rng.choice(len(f_idx), max_entries, replace=False)]
        if len(p_idx) > max_entries: p_idx = p_idx[rng.choice(len(p_idx), max_entries, replace=False)]

    def realized_vec(idx):
        """Realized net return per entry (finite-only) + the bar-rows kept (for regime tagging). Vectorized for 'time'."""
        if exit_kind == "time":
            v = TIMEa[idx[:, 0], idx[:, 1]]
        else:
            v = np.array([realized_path(di, j) for di, j in idx])
        m = np.isfinite(v)
        return v[m], idx[m, 0], MFEa[idx[m, 0], idx[m, 1]]

    real, real_rows, mfe_r = realized_vec(f_idx)
    pool_real, pool_rows, _ = realized_vec(p_idx)
    cap_agg = float(real.sum() / mfe_r.sum()) if mfe_r.sum() != 0 else float("nan")  # HONEST aggregate (not mean-of-ratios)
    nullmeans = np.array([rng.choice(pool_real, size=len(real), replace=False).mean() for _ in range(n_null)])
    p_real = float(np.mean(nullmeans >= real.mean()))    # one-sided: TI realized beats random ENTRY?
    out = {"ti": ti, "tf": tf, "exit": exit_kind, "hold_bars": hold, "n_fired": int(len(real)),
           "mean_MFE_fired": round(100 * float(mfe_r.mean()), 2),
           "mean_realized_net": round(100 * float(real.mean()), 2),
           "null_realized_net": round(100 * float(nullmeans.mean()), 2),
           "edge_vs_random_pp": round(100 * float(real.mean() - nullmeans.mean()), 2),
           "p_vs_random": round(p_real, 4),
           "capture_rate": round(cap_agg, 3)}        # aggregate sum(realized)/sum(MFE) in [.,1]
    if block:                                         # HONEST date-block bootstrap (overlap-aware p)
        out["block"] = block_edge_p(real_rows, real, pool_rows, pool_real, len(C.index), block_bars=21 * bpd)
    if by_regime:                                        # DECISIVE: realized vs random-ENTRY null WITHIN each regime
        ra = reg.to_numpy()
        per = {}
        for rg in ("bull", "chop", "bear"):
            fmask = ra[real_rows] == rg; pmask = ra[pool_rows] == rg
            rr = real[fmask]; pr = pool_real[pmask]
            if len(rr) < 20 or len(pr) < 20: per[rg] = None; continue
            # use replace=True when pool is smaller than sample (bear pool can be thin)
            rep = len(pr) < len(rr)
            nm = np.array([rng.choice(pr, size=len(rr), replace=rep).mean() for _ in range(n_null)])
            d = {"n": int(len(rr)), "realized_net": round(100 * float(rr.mean()), 2),
                 "null_net": round(100 * float(nm.mean()), 2),
                 "edge_pp": round(100 * float(rr.mean() - nm.mean()), 2),
                 "p_vs_random": round(float(np.mean(nm >= rr.mean())), 4)}
            if block:                                    # honest overlap-aware p per regime
                d["block"] = block_edge_p(real_rows[fmask], rr, pool_rows[pmask], pr, len(C.index), block_bars=21 * bpd)
            per[rg] = d
        out["by_regime"] = per
    return out


def selftest(tf="1d"):
    print(f"[selftest] capture_lab tf={tf} -- MOVE-CATCH capture-rate (churn-immune random-ENTRY null), DEV wall <= {fl.DEV_END}")
    lab = fl.load_wide(n=50, tf=tf, min_bars=(200 * fl.BARS_PER_DAY[tf] if tf != "1d" else 400))
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"
    print(f"  {len(lab['syms'])} assets; range {C.index.min()} -> {C.index.max()}")
    print(f"  {'TI':10}{'exit':9}{'nFired':>7}{'MFE%':>8}{'realiz%':>9}{'null%':>8}{'edge_pp':>9}{'p_rnd':>8}{'capture':>9}")
    for ti in ("mom14", "brk14", "rsi14", "volexp", "vpin", "ofi"):
        r = evaluate_ti(lab, ti, tf=tf, exit_kind="time", n_null=200)
        if "note" in r:
            print(f"  {ti:10}{'time':9}{r['n_fired']:>7}  ({r['note']})"); continue
        print(f"  {r['ti']:10}{r['exit']:9}{r['n_fired']:>7}{r['mean_MFE_fired']:>8}{r['mean_realized_net']:>9}"
              f"{r['null_realized_net']:>8}{r['edge_vs_random_pp']:>9}{r['p_vs_random']:>8}{r['capture_rate']:>9}")
    rr = evaluate_ti(lab, "mom14", tf=tf, exit_kind="time", n_null=200, by_regime=True)
    print(f"\n  DECISIVE regime split (mom14, time-exit) -- realized vs random-ENTRY null WITHIN each regime:")
    for rg, d in rr["by_regime"].items():
        if d: print(f"    {rg:6} n={d['n']:>6}  realized {d['realized_net']:>6}%  null {d['null_net']:>6}%  "
                     f"edge {d['edge_pp']:>5}pp  p={d['p_vs_random']}")
    print("[selftest] PASSED -- capture-rate vs random-ENTRY null works, DEV-walled.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    raise SystemExit(selftest(a.tf) if a.selftest else 0)
