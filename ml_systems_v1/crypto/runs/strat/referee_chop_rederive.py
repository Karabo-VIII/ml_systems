"""referee_chop_rederive.py -- INDEPENDENT re-derivation of the load-bearing chop move-CATCH edge.

Adversarial referee (quant). Re-derives, from capture_lab primitives, the mom14/brk14/rsi14 CHOP edge_vs_random
with the FOUR kills, done correctly and consistently, to resolve the lane disagreement on the regime-shuffle:
  (1) DATE-BLOCK moving-block bootstrap on the EDGE statistic (resample whole calendar dates, not iid entries) -> honest p, p05.
  (2) REVERSE-SCORE: do BOTTOM-momentum entries LOSE in chop (direction-sensitive, not concentration)?
  (3) REGIME-LABEL shuffle done TWO honest ways:
        (3a) circular block-ROTATION of the regime label time-series (preserves composition + autocorr; breaks TI<->regime align)
        (3b) the edge recomputed with rotated labels uses the SAME rotated labels for BOTH the TI-fired set AND the same-regime pool
             (this is the fix for the product_sweep complaint: the pool is re-drawn under the shuffled labels too, so no bull-bar injection)
  (4) CALENDAR-invariance: edge/day at 1d vs 4h vs 1h.

Also reports n_eff (unique fired dates) to expose the iid SE deflation.
DEV-walled. Long-only spot, taker cost. No emoji. RWYB.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))
import strat.fleet_lab as fl
import strat.capture_lab as cl

RNG = np.random.default_rng(12345)


def fired_pool_arrays(lab, ti, tf, hold, min_move, warm=40):
    """Return per-entry realized net (time-exit), the date-row index, and the regime label, for FIRED set and POOL.
    Mirrors capture_lab.evaluate_ti's universe exactly (time-exit, vectorized)."""
    C = lab["C"]
    MFE = cl.mfe_matrix(C, hold).to_numpy()
    TIME = cl.time_return_matrix(C, hold).to_numpy()
    Ca = C.to_numpy()
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), dtype=bool)
    valid[warm:n - hold - 1, :] = True
    valid &= np.isfinite(Ca) & np.isfinite(MFE)
    fired = cl.fired_matrix(lab, ti).to_numpy()
    fmat = fired & valid & (MFE > min_move)
    poolmat = valid & (MFE > min_move)
    fi = np.array(np.where(fmat)).T            # (n,2) rows=date_idx, cols=asset
    pi = np.array(np.where(poolmat)).T
    # time-exit realized net is just TIME[date,asset]
    fr = TIME[fi[:, 0], fi[:, 1]]; mf = np.isfinite(fr)
    fi, fr = fi[mf], fr[mf]
    pr = TIME[pi[:, 0], pi[:, 1]]; mp = np.isfinite(pr)
    pi, pr = pi[mp], pr[mp]
    return fi, fr, pi, pr   # fi/pi: row index into C.index (date)


def block_bootstrap_edge(fi, fr, pi, pr, reg_arr, regime, dates, block_days=21, n_boot=2000, seed=1):
    """Honest moving-block bootstrap of the WITHIN-REGIME edge (TI-fired realized - same-regime-pool realized).
    Resamples by CALENDAR DATE blocks (not iid entries): pick contiguous date-blocks, take all entries whose
    date falls in a sampled block. This respects the heavy overlap (7d holds, 50 assets share dates)."""
    rng = np.random.default_rng(seed)
    # restrict to the regime
    fm = reg_arr[fi[:, 0]] == regime
    pm = reg_arr[pi[:, 0]] == regime
    fr_r, fdate = fr[fm], fi[fm, 0]
    pr_r, pdate = pr[pm], pi[pm, 0]
    if len(fr_r) < 20 or len(pr_r) < 20:
        return None
    obs = fr_r.mean() - pr_r.mean()
    n_dates = len(dates)
    # group entries by date for fast block assembly
    # map: for each date-row, the realized values present (fired and pool)
    from collections import defaultdict
    fd = defaultdict(list); pd_ = defaultdict(list)
    for v, d in zip(fr_r, fdate): fd[d].append(v)
    for v, d in zip(pr_r, pdate): pd_[d].append(v)
    all_dates = np.arange(n_dates)
    n_blocks = max(1, n_dates // block_days)
    boots = np.empty(n_boot)
    nb = 0
    for b in range(n_boot):
        starts = rng.integers(0, n_dates - block_days + 1, size=n_blocks)
        sel = np.concatenate([np.arange(s, s + block_days) for s in starts])
        fvals = []; pvals = []
        for d in sel:
            if d in fd: fvals.extend(fd[d])
            if d in pd_: pvals.extend(pd_[d])
        if len(fvals) < 5 or len(pvals) < 5:
            boots[b] = np.nan; continue
        boots[b] = np.mean(fvals) - np.mean(pvals); nb += 1
    boots = boots[np.isfinite(boots)]
    # one-sided p that the edge is <= 0 (null), via bootstrap centered at 0
    centered = boots - boots.mean()
    p_block = float(np.mean(centered + 0.0 >= obs)) if obs > 0 else float(np.mean(centered <= obs))
    # cleaner: fraction of bootstrap edges <= 0 (how often the resampled edge is non-positive)
    p_le0 = float(np.mean(boots <= 0.0))
    return {
        "regime": regime, "n_fired": int(len(fr_r)), "n_pool": int(len(pr_r)),
        "n_eff_fired_dates": int(len(np.unique(fdate))),
        "obs_edge_pp": round(100 * obs, 3),
        "boot_p05_pp": round(100 * float(np.percentile(boots, 5)), 3),
        "boot_p50_pp": round(100 * float(np.percentile(boots, 50)), 3),
        "boot_mean_pp": round(100 * float(boots.mean()), 3),
        "p_block_le0": round(p_le0, 4),
        "n_boot_valid": int(len(boots)),
    }


def reverse_score(lab, ti, tf, hold, min_move, reg_arr, regime, warm=40):
    """Direction test: BOTTOM-tercile of the TI score should LOSE in this regime if the edge is directional."""
    C = lab["C"]; F = lab["F"]; X = F[ti].to_numpy()
    MFE = cl.mfe_matrix(C, hold).to_numpy(); TIME = cl.time_return_matrix(C, hold).to_numpy(); Ca = C.to_numpy()
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), dtype=bool)
    valid[warm:n - hold - 1, :] = True
    valid &= np.isfinite(Ca) & np.isfinite(MFE) & (MFE > min_move) & np.isfinite(X)
    # cross-sectional terciles per date among valid; bottom = anti-signal
    fired_top = cl.fired_matrix(lab, ti).to_numpy() & valid
    # bottom: invert the rule -> strongly negative momentum / low rsi / low rangepos / low brk
    if ti in ("mom7", "mom14", "mom30", "accel", "brk14"):
        bottom = (X < 0) & valid
    elif ti == "rsi14":
        bottom = (X < 45) & valid
    elif ti == "rangepos":
        bottom = (X < 0.3) & valid
    elif ti == "volexp":
        bottom = (X < 0.8) & valid
    else:
        bottom = valid & ~fired_top
    def reg_mean(mask):
        idx = np.array(np.where(mask & (reg_arr[:, None] == regime))).T
        if len(idx) < 20: return np.nan, 0
        v = TIME[idx[:, 0], idx[:, 1]]; v = v[np.isfinite(v)]
        return float(v.mean()), len(v)
    top_m, top_n = reg_mean(fired_top)
    bot_m, bot_n = reg_mean(bottom)
    return {"regime": regime, "top_realized_pp": round(100 * top_m, 2) if top_n else None, "top_n": top_n,
            "bottom_realized_pp": round(100 * bot_m, 2) if bot_n else None, "bottom_n": bot_n,
            "direction_ok": bool(top_n and bot_n and top_m > bot_m)}


def regime_shuffle(lab, ti, tf, hold, min_move, reg_arr, regime, n_shuf=500, block_days=21, seed=7, warm=40):
    """HONEST regime-label shuffle: circularly ROTATE the regime-label time series by a random offset (preserves
    composition + label autocorrelation, breaks TI<->regime alignment). CRUCIALLY: the same-regime POOL is re-drawn
    under the rotated labels too (fixes the product_sweep 'bull-bar injection' complaint -- both fired & pool use the
    SAME rotated labels). If the chop edge is regime-conditional, real_edge should sit in the RIGHT tail of the
    rotated-label edge distribution (real >> shuffled). If real sits in the middle/left, it is NOT regime-conditional."""
    rng = np.random.default_rng(seed)
    fi, fr, pi, pr = fired_pool_arrays(lab, ti, tf, hold, min_move, warm)
    n = len(reg_arr)
    # observed edge under TRUE labels
    fm = reg_arr[fi[:, 0]] == regime; pm = reg_arr[pi[:, 0]] == regime
    if fm.sum() < 20 or pm.sum() < 20:
        return None
    obs = fr[fm].mean() - pr[pm].mean()
    shuf_edges = []
    for _ in range(n_shuf):
        # rotate labels by a random offset >= block_days (avoid trivial near-identity)
        off = int(rng.integers(block_days, n - block_days))
        rr = np.roll(reg_arr, off)
        fmm = rr[fi[:, 0]] == regime; pmm = rr[pi[:, 0]] == regime
        if fmm.sum() < 20 or pmm.sum() < 20: continue
        shuf_edges.append(fr[fmm].mean() - pr[pmm].mean())
    shuf_edges = np.array(shuf_edges)
    p_ge = float(np.mean(shuf_edges >= obs))   # how often shuffle >= real; high => real NOT special
    return {"regime": regime, "obs_edge_pp": round(100 * obs, 3),
            "shuf_mean_pp": round(100 * float(shuf_edges.mean()), 3),
            "shuf_p95_pp": round(100 * float(np.percentile(shuf_edges, 95)), 3),
            "p_shuf_ge_real": round(p_ge, 4), "n_shuf": int(len(shuf_edges)),
            "regime_conditional": bool(p_ge < 0.05)}


def run_tf(tf, tis, hold_days=7, min_move=0.03, regimes=("bull", "chop", "bear")):
    bpd = fl.BARS_PER_DAY[tf]; hold = hold_days * bpd
    lab = fl.load_wide(n=50, tf=tf, min_bars=(200 * bpd if tf != "1d" else 400))
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"
    reg = cl.regime_series(lab, tf); reg_arr = reg.to_numpy(); dates = C.index
    out = {"tf": tf, "hold_bars": hold, "date_max": str(C.index.max()), "n_assets": len(lab["syms"]), "tis": {}}
    for ti in tis:
        fi, fr, pi, pr = fired_pool_arrays(lab, ti, tf, hold, min_move)
        ti_out = {"regimes": {}}
        for rg in regimes:
            boot = block_bootstrap_edge(fi, fr, pi, pr, reg_arr, rg, dates,
                                        block_days=max(3, 21 // bpd) if tf != "1d" else 21,
                                        n_boot=2000, seed=1)
            rev = reverse_score(lab, ti, tf, hold, min_move, reg_arr, rg)
            shu = regime_shuffle(lab, ti, tf, hold, min_move, reg_arr, rg, n_shuf=500,
                                 block_days=max(3, 21 // bpd) if tf != "1d" else 21)
            ti_out["regimes"][rg] = {"block_boot": boot, "reverse": rev, "regime_shuffle": shu}
        out["tis"][ti] = ti_out
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="1d,4h,1h")
    ap.add_argument("--tis", default="mom14,brk14,rsi14")
    a = ap.parse_args()
    tis = a.tis.split(",")
    results = {}
    for tf in a.tfs.split(","):
        print(f"\n===== TF={tf} =====", flush=True)
        r = run_tf(tf, tis)
        results[tf] = r
        for ti, d in r["tis"].items():
            for rg, dd in d["regimes"].items():
                b = dd["block_boot"]; rev = dd["reverse"]; sh = dd["regime_shuffle"]
                if b is None: print(f"  {ti:7} {rg:5}  (insufficient)"); continue
                print(f"  {ti:7} {rg:5} | obs={b['obs_edge_pp']:+6.2f}pp p05={b['boot_p05_pp']:+6.2f} "
                      f"p_block_le0={b['p_block_le0']:.3f} n_eff={b['n_eff_fired_dates']:>4} | "
                      f"rev: top={rev['top_realized_pp']} bot={rev['bottom_realized_pp']} dir_ok={rev['direction_ok']} | "
                      f"shuf: obs={sh['obs_edge_pp']:+.2f} shufmean={sh['shuf_mean_pp']:+.2f} "
                      f"p_shuf>=real={sh['p_shuf_ge_real']:.3f} regime_cond={sh['regime_conditional']}", flush=True)
    outp = Path(__file__).resolve().parent / "referee_chop_rederive_results.json"
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {outp}")
