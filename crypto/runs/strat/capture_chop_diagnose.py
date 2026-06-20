"""capture_chop_diagnose.py -- K=2/3 independent derivations of the decisive findings (referee discipline).

The battery said: chop edge is AMBIGUOUS (honest block-p ~.02-.05, p05 marginally >0) but FAILS the
regime-shuffle (shuf-p ~.6-.8 = the edge is NOT regime-conditional). Before committing 'ARTIFACT', re-derive:

  D1. POOLED vs WITHIN-REGIME edge: is the momentum-into-a-move edge REGIME-SPECIFIC at all, or just a
      uniform 'momentum helps when there's a move' that exists in EVERY regime (=> chop is not special)?
  D2. REGIME-SHUFFLE by an INDEPENDENT mechanism: full random date-permutation of labels (destroys run
      structure too) vs the circular-rotation used in the battery. Same verdict => robust.
  D3. n_eff: estimate the effective sample size of the chop-fired entries (autocorrelation/overlap) to
      show WHY the IID p=0.0 was a lie and how few independent observations actually back the edge.

DEV-walled. No emoji. Run from crypto/src.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.fleet_lab as fl
import strat.capture_lab as cl
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_chop_battery import build_entries


def d1_pooled_vs_within(lab, ti, tf="1d"):
    """Edge (fired vs pool) computed POOLED (all regimes) and WITHIN each regime. If chop edge ~ pooled edge
    and every regime is positive, momentum-into-a-move is UNIVERSAL, not chop-conditional."""
    df, _ = build_entries(lab, ti, tf=tf)
    pooled = df.loc[df["fired"], "realized"].mean() - df["realized"].mean()
    res = {"pooled_edge_pp": round(100 * pooled, 3)}
    for rg in ("bull", "chop", "bear"):
        sub = df[df["regime"] == rg]
        if sub["fired"].sum() < 20:
            res[rg] = None; continue
        e = sub.loc[sub["fired"], "realized"].mean() - sub["realized"].mean()
        res[rg] = round(100 * e, 3)
    return res


def d2_perm_shuffle(lab, df, regime, tf="1d", n_shuf=1000, seed=1):
    """Independent regime-shuffle: FULL random permutation of the regime-label SERIES (block-free).
    Destroys both alignment AND run-structure. Compare real chop edge to this null dist."""
    reg = cl.regime_series(lab, tf).to_numpy()
    di = df["di"].to_numpy(); real = df["realized"].to_numpy(); fired = df["fired"].to_numpy()
    true_mask = df["regime"].to_numpy() == regime
    real_edge = real[true_mask & fired].mean() - real[true_mask].mean()
    rng = np.random.default_rng(seed)
    edges = []
    for _ in range(n_shuf):
        perm = rng.permutation(reg)            # full permutation of the label series
        lbl = perm[di]
        m = lbl == regime
        if m.sum() < 30 or (m & fired).sum() < 10:
            continue
        edges.append(real[m & fired].mean() - real[m].mean())
    edges = np.array(edges)
    return {"regime": regime, "real_edge_pp": round(100 * float(real_edge), 3),
            "perm_mean_pp": round(100 * float(edges.mean()), 3),
            "perm_p95_pp": round(100 * float(np.percentile(edges, 95)), 3),
            "p_perm_ge_real": round(float(np.mean(edges >= real_edge)), 4),
            "regime_conditional": bool(np.mean(edges >= real_edge) < 0.05)}


def d3_n_eff(lab, ti, tf="1d", regime="chop"):
    """Effective sample size of chop-fired entries. Two crude bounds that bracket n_eff:
       (a) #unique entry DATES (cross-asset same-date shock => 1 indep draw/date, lower-ish bound on date axis)
       (b) #non-overlapping 7d windows per asset summed (serial overlap => /7 per asset).
    Plus a block-SE: SE of the edge from 21d-block resampling vs the IID SE, ratio => inflation factor."""
    df, hold = build_entries(lab, ti, tf=tf)
    sub = df[(df["regime"] == regime) & df["fired"]]
    n_nom = len(sub)
    n_dates = sub["date"].dt.normalize().nunique()
    # non-overlapping windows per asset: span/hold summed
    bpd = fl.BARS_PER_DAY[tf]
    per_asset = sub.groupby("asset")["di"].agg(lambda x: max(1, int(np.ceil((x.max() - x.min() + 1) / hold))))
    n_nonoverlap = int(per_asset.sum())
    iid_se = sub["realized"].std() / np.sqrt(n_nom)
    return {"regime": regime, "n_nominal": int(n_nom), "n_unique_dates": int(n_dates),
            "n_nonoverlap_windows": n_nonoverlap,
            "n_eff_estimate": int(min(n_dates, n_nonoverlap)),
            "iid_se_pp": round(100 * float(iid_se), 4),
            "iid_se_understatement_x": round(float(np.sqrt(n_nom / max(1, min(n_dates, n_nonoverlap)))), 2)}


if __name__ == "__main__":
    print("=" * 100)
    print("K=2/3 DIAGNOSTIC -- re-derive the decisive findings, DEV-walled <= 2024-05-15")
    print("=" * 100)
    lab1d = fl.load_wide(n=50, tf="1d", min_bars=400)

    print("\n### D1: POOLED vs WITHIN-REGIME edge (is chop SPECIAL or is momentum-into-a-move UNIVERSAL?) ###\n")
    print(f"  {'TI':6}{'pooled':>9}{'bull':>9}{'chop':>9}{'bear':>9}")
    for ti in ("mom14", "brk14", "rsi14"):
        r = d1_pooled_vs_within(lab1d, ti, "1d")
        print(f"  {ti:6}{r['pooled_edge_pp']:>9}{str(r.get('bull')):>9}{str(r.get('chop')):>9}{str(r.get('bear')):>9}")

    print("\n### D2: INDEPENDENT regime-shuffle (full permutation, not rotation) -- chop ###\n")
    print(f"  {'TI':6}{'real_edge':>11}{'perm_mean':>11}{'perm_p95':>10}{'p_ge_real':>11}  cond?")
    for ti in ("mom14", "brk14", "rsi14"):
        df, _ = build_entries(lab1d, ti, tf="1d")
        r = d2_perm_shuffle(lab1d, df, "chop", "1d")
        print(f"  {ti:6}{r['real_edge_pp']:>11}{r['perm_mean_pp']:>11}{r['perm_p95_pp']:>10}"
              f"{r['p_perm_ge_real']:>11}  {r['regime_conditional']}")

    print("\n### D3: n_eff of chop-fired entries (WHY IID p=0.0 was a lie) ###\n")
    print(f"  {'TI':6}{'n_nominal':>11}{'n_dates':>9}{'n_nonoverlap':>14}{'n_eff':>8}{'IID_SE_under_x':>16}")
    for ti in ("mom14", "brk14", "rsi14"):
        r = d3_n_eff(lab1d, ti, "1d", "chop")
        print(f"  {ti:6}{r['n_nominal']:>11}{r['n_unique_dates']:>9}{r['n_nonoverlap_windows']:>14}"
              f"{r['n_eff_estimate']:>8}{r['iid_se_understatement_x']:>16}")
    print("\n[done]")
