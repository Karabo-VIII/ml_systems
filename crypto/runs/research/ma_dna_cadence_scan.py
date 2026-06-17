"""runs/research/ma_dna_cadence_scan.py

BROAD CHEAP SCAN: which bar-type / timeframe carries the strongest MA-DNA signal for ORACLE entries?

WHAT
----
Oracle-decomposition step 2 (docs/ORACLE_DECOMPOSITION_2026_06_06.md): label each bar oracle-ENTRY
(from the audited perfect-foresight high-capture DP in oracle_ceiling_builder.py) vs non, then learn
P(oracle-entry | MA-based causal features), held-out, with a SHUFFLED-LABEL control + L2.

This scan differs from oracle_dna_shuffled_falsifier.py (which used the 40 canonical chimera norm_/xd_
features): here the feature BASIS is MA-decomposition ONLY, grouped so we can read WHICH part of the
MA story (if any) carries oracle-entry information, and across MANY cadences/bar-types so we can rank
(cadence x MA-decomposition) by held-out predictive lift over the shuffled control.

MA-DECOMPOSITION GROUPS (all strictly past-only: every feature at bar i uses close[i-1] and earlier):
  single   : single-MA distance + slope            (dist20, slope20, dist50)
  two_ma   : 2-MA gap + cross-STATE (not the event) (gap_10_30, crossstate_10_30, gap_20_50)
  ribbon3  : 3-MA ribbon order + compression        (order_score, compression, ribbon_dist)
  ma_of_ma : MA-of-MA (double-smoothed)             (mama_dist, mama_slope, ma_vs_mama)
  kama_er  : Kaufman efficiency-ratio + KAMA        (er10, er20, kama_dist)
  ALL      : concat of all the above (the full MA-DNA basis, L2-regularized)

NOT re-mining MA-CROSS as a raw trigger (already REFUTED): cross-STATE enters only as one regularized
feature inside a learned model -- decomposition basis, never a standalone entry rule.

METRIC (the rank key): held-out (OOS+UNSEEN) ROC-AUC of P(oracle-entry|MA feats), and its LIFT over the
shuffled-label control (real_auc - shuffled_mean_auc). A group "carries MA-DNA" iff real_auc > shuffled
p95 by a margin. Secondary (honesty): forward-return Spearman IC of the predicted prob -- AUC-lift on
oracle-entry CLASSIFICATION does NOT imply tradeable skill (the SOL-4h full-feature run already showed
AUC~0.64 with ~0 capture skill); IC keeps that distinction visible.

HARD CONSTRAINTS: LONG-ONLY, past-only features (close.shift(1)), fit on TRAIN+VAL, evaluate on held-out
OOS+UNSEEN the model never saw, shuffle permutes FIT labels only, L2 logistic (C=0.5, balanced).

RWYB:
    .venv/Scripts/python.exe runs/research/ma_dna_cadence_scan.py --selftest
    .venv/Scripts/python.exe runs/research/ma_dna_cadence_scan.py            # full scan
    .venv/Scripts/python.exe runs/research/ma_dna_cadence_scan.py --quick    # fewer cadences/shuffles
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs" / "research"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from numba import njit  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from oracle_ceiling_builder import oracle_high_capture, WIN  # noqa: E402  (reuse audited oracle DP)

COST_RT = 0.0024

__contract__ = {
    "kind": "ma_dna_cadence_scan",
    "version": "1.0",
    "inputs": ["chimera (asset, cadence) open/high/close + ts", "oracle-entry labels from oracle_high_capture"],
    "outputs": ["per (asset,cadence,MA-group): held-out AUC, shuffled-null AUC dist, lift, fwd-IC; ranked"],
    "invariants": [
        "MA features strictly past-only (close.shift(1) base); NO raw OHLC/target/forward cols in X",
        "fit on TRAIN+VAL, evaluate on held-out OOS+UNSEEN; shuffle permutes FIT labels only",
        "cross-STATE used as a regularized feature only -- never a standalone MA-cross trigger (REFUTED)",
        "L2 logistic C=0.5 class_weight=balanced; rank key = real_auc - shuffled_mean_auc on held-out",
        "AUC-lift is CLASSIFICATION lift, not tradeable skill -- fwd-IC reported alongside for honesty",
    ],
}


# ===========================================================================
# MA-DECOMPOSITION FEATURE GROUPS  (strictly past-only)
# ===========================================================================
@njit(cache=True)
def _kama(px, er, fast=2.0, slow=30.0):
    """Kaufman adaptive MA. px = past-only last close (already shifted). er = efficiency ratio in [0,1]."""
    n = px.shape[0]
    out = np.full(n, np.nan)
    fsc = 2.0 / (fast + 1.0)
    ssc = 2.0 / (slow + 1.0)
    prev = np.nan
    for i in range(n):
        p = px[i]
        e = er[i]
        if np.isnan(p) or np.isnan(e):
            continue
        sc = (e * (fsc - ssc) + ssc) ** 2
        if np.isnan(prev):
            prev = p
        else:
            prev = prev + sc * (p - prev)
        out[i] = prev
    return out


def build_ma_groups(close):
    """Return {group_name: (feat_names, X[n,k])}. Every feature uses close[i-1] and earlier only."""
    s = pd.Series(close.astype(np.float64))
    px = s.shift(1)                                  # last KNOWN close at decision time for bar i

    def sma(w):
        return px.rolling(w, min_periods=w).mean()

    def ema(span):
        return px.ewm(span=span, adjust=False, min_periods=span).mean()

    def slope(ma, k=3):
        return ma / ma.shift(k) - 1.0

    # --- single-MA: distance + slope
    sma20, sma50 = sma(20), sma(50)
    single = {
        "dist20": (px / sma20 - 1.0),
        "slope20": slope(sma20, 3),
        "dist50": (px / sma50 - 1.0),
    }

    # --- 2-MA: gap magnitude + cross-STATE (sign), at two scales
    ma10, ma30 = sma(10), sma(30)
    two_ma = {
        "gap_10_30": (ma10 - ma30) / ma30,
        "crossstate_10_30": np.sign(ma10 - ma30),
        "gap_20_50": (sma20 - sma50) / sma50,
    }

    # --- 3-MA ribbon: order score + compression (width/price) + distance from center
    mf, mm, ms = sma(10), sma(20), sma(40)
    order_score = np.sign(mf - mm) + np.sign(mm - ms)           # in [-2..2]; +2 = fully bullish ribbon
    width = (pd.concat([mf, mm, ms], axis=1).max(axis=1)
             - pd.concat([mf, mm, ms], axis=1).min(axis=1))
    ribbon3 = {
        "order_score": order_score,
        "compression": width / px,
        "ribbon_dist": px / mm - 1.0,
    }

    # --- MA-of-MA (double smoothed): distance + slope + first-MA vs double-MA
    mama = sma(10).rolling(10, min_periods=10).mean()           # SMA10 of SMA10 (past-only twice)
    ma_of_ma = {
        "mama_dist": px / mama - 1.0,
        "mama_slope": slope(mama, 3),
        "ma_vs_mama": ma10 / mama - 1.0,
    }

    # --- KAMA / efficiency ratio (Kaufman)
    def er(n):
        direction = (px - px.shift(n)).abs()
        volatility = px.diff().abs().rolling(n, min_periods=n).sum()
        return direction / volatility
    er10, er20 = er(10), er(20)
    kama = _kama(px.to_numpy(), np.nan_to_num(er10.to_numpy(), nan=0.0))
    kama_er = {
        "er10": er10,
        "er20": er20,
        "kama_dist": px / pd.Series(kama) - 1.0,
    }

    groups = {"single": single, "two_ma": two_ma, "ribbon3": ribbon3,
              "ma_of_ma": ma_of_ma, "kama_er": kama_er}

    out = {}
    allnames, allcols = [], []
    for gname, gd in groups.items():
        names = list(gd.keys())
        cols = np.column_stack([np.asarray(gd[k], dtype=np.float64) for k in names])
        out[gname] = (names, cols)
        allnames += [f"{gname}.{k}" for k in names]
        allcols.append(cols)
    out["ALL"] = (allnames, np.column_stack(allcols))
    return out


# ===========================================================================
def _window_mask(ts_ms, name):
    lo, hi = WIN[name]
    lo_ms = 0 if lo == "0" else int(pd.Timestamp(lo).value // 1_000_000)
    hi_ms = int(pd.Timestamp(hi).value // 1_000_000)
    return (ts_ms >= lo_ms) & (ts_ms < hi_ms)


def fwd_open_to_open(op, H, cost=COST_RT):
    """Honest net return of entering at open[i+1], exiting at open[i+1+H]."""
    n = len(op)
    fwd = np.full(n, np.nan)
    for i in range(n):
        ei, xi = i + 1, i + 1 + H
        if xi < n and ei < n:
            fwd[i] = op[xi] / op[ei] - 1.0 - cost
    return fwd


def load_ohlc(asset, cadence):
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(asset + "USDT", cadence=cadence)
    ts = g["timestamp"].to_numpy().astype(np.int64)
    op = g["open"].to_numpy().astype(np.float64)
    hi = g["high"].to_numpy().astype(np.float64)
    cl = g["close"].to_numpy().astype(np.float64)
    if not np.all(np.diff(ts) > 0):
        order = np.argsort(ts, kind="stable")
        ts, op, hi, cl = ts[order], op[order], hi[order], cl[order]
    return ts, op, hi, cl


def fit_eval_group(X, y, fwd, m_fit, m_ho, n_shuffle, seed=7):
    """Fit L2 logistic on m_fit, eval held-out AUC + fwd-IC; shuffled-label control over n_shuffle seeds.
    Returns dict or None if the group has no valid rows / single-class fit."""
    valid = np.all(np.isfinite(X), axis=1)
    fit_idx = np.where(m_fit & valid)[0]
    ho_idx = np.where(m_ho & valid)[0]
    if len(fit_idx) < 200 or len(ho_idx) < 100:
        return None
    Xtr, ytr = X[fit_idx], y[fit_idx]
    Xho, yho = X[ho_idx], y[ho_idx]
    fho = fwd[ho_idx]
    if len(np.unique(ytr)) < 2 or len(np.unique(yho)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xho_s = sc.transform(Xtr), sc.transform(Xho)

    def _fit_predict(yfit):
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr_s, yfit)
        return clf.predict_proba(Xho_s)[:, 1]

    p_real = _fit_predict(ytr)
    real_auc = float(roc_auc_score(yho, p_real))
    mm = np.isfinite(fho) & np.isfinite(p_real)
    real_ic = float(spearmanr(p_real[mm], fho[mm])[0]) if mm.sum() > 20 else float("nan")

    sh = []
    for s in range(n_shuffle):
        rng = np.random.default_rng(1000 + s)
        ys = ytr[rng.permutation(len(ytr))]
        if len(np.unique(ys)) < 2:
            continue
        sh.append(float(roc_auc_score(yho, _fit_predict(ys))))
    sh = np.asarray(sh, float)
    sh_mean = float(sh.mean()) if len(sh) else float("nan")
    sh_p95 = float(np.percentile(sh, 95)) if len(sh) else float("nan")
    return {
        "real_auc": real_auc, "shuffled_mean_auc": sh_mean, "shuffled_p95_auc": sh_p95,
        "auc_lift": real_auc - sh_mean, "beats_shuffled_p95": bool(real_auc > sh_p95),
        "fwd_ic": real_ic, "n_fit": int(len(fit_idx)), "n_ho": int(len(ho_idx)),
        "base_rate_fit": float(ytr.mean()),
    }


def run_one(asset, cadence, n_shuffle, exit_h=None, verbose=True):
    ts, op, hi, cl = load_ohlc(asset, cadence)
    n = len(op)
    f_dp, trades = oracle_high_capture(ts, op, hi)
    y = np.zeros(n, dtype=int)
    if not trades:
        return {"asset": asset, "cadence": cadence, "error": "no oracle trades"}
    ent = np.array([i for i, j in trades], dtype=int)
    y[ent] = 1
    holds = np.array([j - i for i, j in trades], dtype=int)
    H = int(exit_h) if exit_h else int(max(1, np.median(holds)))
    fwd = fwd_open_to_open(op, H)

    groups = build_ma_groups(cl)
    m_fit = _window_mask(ts, "TRAIN") | _window_mask(ts, "VAL")
    m_ho = _window_mask(ts, "OOS") | _window_mask(ts, "UNSEEN")

    out = {"asset": asset, "cadence": cadence, "n_bars": int(n),
           "oracle_entries": int(y.sum()), "oracle_base_rate": float(y.mean()),
           "exit_H_bars": int(H), "n_fit_rows": int(m_fit.sum()), "n_ho_rows": int(m_ho.sum()),
           "groups": {}}
    for gname, (names, X) in groups.items():
        r = fit_eval_group(X, y, fwd, m_fit, m_ho, n_shuffle)
        if r is not None:
            r["features"] = names
        out["groups"][gname] = r
    if verbose:
        print(f"\n{asset} {cadence}: bars={n} oracle_entries={int(y.sum())} "
              f"({y.mean():.3f}) exit_H={H}  fit={int(m_fit.sum())} ho={int(m_ho.sum())}")
        for gname in ["single", "two_ma", "ribbon3", "ma_of_ma", "kama_er", "ALL"]:
            r = out["groups"].get(gname)
            if r is None:
                print(f"    {gname:9} --skip--")
                continue
            star = "*" if r["beats_shuffled_p95"] else " "
            print(f"   {star}{gname:9} AUC={r['real_auc']:.4f}  shuf(mean/p95)="
                  f"{r['shuffled_mean_auc']:.4f}/{r['shuffled_p95_auc']:.4f}  "
                  f"lift={r['auc_lift']:+.4f}  fwd_IC={r['fwd_ic']:+.4f}")
    return out


def run_scan(assets, cadences, n_shuffle, exit_h=None):
    results = []
    for cad in cadences:
        for a in assets:
            try:
                results.append(run_one(a, cad, n_shuffle, exit_h=exit_h))
            except Exception as e:
                print(f"   {a} {cad} ERROR {type(e).__name__}: {str(e)[:80]}")
                results.append({"asset": a, "cadence": cad, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    # ---- RANKING: aggregate lift per (cadence x group) across assets (mean of valid)
    agg = {}
    for r in results:
        if "groups" not in r:
            continue
        cad = r["cadence"]
        for gname, gr in r["groups"].items():
            if gr is None:
                continue
            key = (cad, gname)
            agg.setdefault(key, {"lifts": [], "aucs": [], "ics": [], "beats": 0, "n": 0})
            agg[key]["lifts"].append(gr["auc_lift"])
            agg[key]["aucs"].append(gr["real_auc"])
            agg[key]["ics"].append(gr["fwd_ic"])
            agg[key]["beats"] += int(gr["beats_shuffled_p95"])
            agg[key]["n"] += 1
    ranked = []
    for (cad, gname), d in agg.items():
        ranked.append({
            "cadence": cad, "group": gname, "n_assets": d["n"],
            "mean_auc": float(np.nanmean(d["aucs"])),
            "mean_lift": float(np.nanmean(d["lifts"])),
            "mean_fwd_ic": float(np.nanmean(d["ics"])),
            "n_beats_p95": d["beats"],
        })
    ranked.sort(key=lambda x: x["mean_lift"], reverse=True)

    # ---- per-cadence best group (which TIMEFRAME/BAR-TYPE carries the strongest MA-DNA)
    by_cad = {}
    for row in ranked:
        c = row["cadence"]
        if c not in by_cad or row["mean_lift"] > by_cad[c]["mean_lift"]:
            by_cad[c] = row
    cad_rank = sorted(by_cad.values(), key=lambda x: x["mean_lift"], reverse=True)

    return {"per_run": results, "ranked_cadence_x_group": ranked, "best_group_per_cadence": cad_rank}


# ===========================================================================
def _selftest():
    print("=" * 70)
    print("[ma_dna_cadence_scan selftest]")
    print("=" * 70)
    ok = True
    rng = np.random.default_rng(0)
    n = 6000
    # synthetic price; learnable label from a real past-only MA feature -> AUC>0.5, shuffled ~0.5
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    groups = build_ma_groups(close)
    # all MA features must be strictly past-only: feature[i] independent of close[i].
    # perturb close[i] only and confirm feature row i unchanged (look-ahead falsifier).
    c2 = close.copy(); c2[3000] *= 1.5
    g2 = build_ma_groups(c2)
    Xa = groups["ALL"][1][3000]
    Xb = g2["ALL"][1][3000]
    same = np.allclose(np.nan_to_num(Xa), np.nan_to_num(Xb))
    print(f"  past-only check: bumping close[3000] leaves feature row 3000 unchanged = {same} (expect True)")
    ok &= same
    # and it DOES change the NEXT row (uses close[3000] via shift)
    changed_next = not np.allclose(np.nan_to_num(groups["ALL"][1][3001]),
                                   np.nan_to_num(g2["ALL"][1][3001]))
    print(f"  causality check: row 3001 DOES change = {changed_next} (expect True)")
    ok &= changed_next
    # power: learnable label from feature single.dist20 -> AUC>0.6, shuffled collapses
    names, X = groups["single"]
    feat = X[:, 0]
    valid = np.isfinite(feat)
    y = np.zeros(n, int)
    thr = np.nanmedian(feat)
    y[valid] = (feat[valid] + 0.2 * rng.normal(0, np.nanstd(feat[valid]), valid.sum()) > thr).astype(int)
    fwd = rng.normal(0, 0.02, n)
    m_fit = np.zeros(n, bool); m_fit[:4000] = True
    m_ho = np.zeros(n, bool); m_ho[4000:] = True
    r = fit_eval_group(X, y, fwd, m_fit, m_ho, n_shuffle=20)
    print(f"  power: real_auc={r['real_auc']:.3f} (expect>0.6)  shuffled_mean={r['shuffled_mean_auc']:.3f} (expect~0.5)")
    ok &= r["real_auc"] > 0.6 and abs(r["shuffled_mean_auc"] - 0.5) < 0.07
    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,SOL,ETH")
    ap.add_argument("--cadences", default="15m,1h,4h,1d,range,dib")
    ap.add_argument("--n-shuffle", type=int, default=30)
    ap.add_argument("--exit-h", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not _selftest():
        print("SELFTEST FAILED -- aborting before touching market data")
        sys.exit(1)
    assets = args.assets.split(",")
    cadences = args.cadences.split(",")
    n_shuffle = args.n_shuffle
    if args.quick:
        cadences = ["4h", "1d"]
        n_shuffle = 15
    print("\n" + "=" * 80)
    print(f"MA-DNA CADENCE SCAN  assets={assets} cadences={cadences} n_shuffle={n_shuffle}")
    print("=" * 80)
    res = run_scan(assets, cadences, n_shuffle, exit_h=(args.exit_h or None))

    print("\n" + "=" * 80)
    print("RANKED (cadence x MA-group) by mean held-out AUC-lift over shuffled control")
    print("=" * 80)
    print(f"  {'cadence':8} {'group':9} {'n':>2}  {'mean_auc':>8} {'mean_lift':>9} {'beats_p95':>9} {'fwd_IC':>8}")
    for row in res["ranked_cadence_x_group"]:
        print(f"  {row['cadence']:8} {row['group']:9} {row['n_assets']:>2}  "
              f"{row['mean_auc']:>8.4f} {row['mean_lift']:>+9.4f} "
              f"{row['n_beats_p95']:>4}/{row['n_assets']:<4} {row['mean_fwd_ic']:>+8.4f}")
    print("\n" + "-" * 80)
    print("BEST MA-group per cadence (which bar-type/timeframe carries strongest MA-DNA)")
    print("-" * 80)
    for row in res["best_group_per_cadence"]:
        print(f"  {row['cadence']:8} best={row['group']:9} mean_lift={row['mean_lift']:+.4f}  "
              f"mean_auc={row['mean_auc']:.4f}  fwd_IC={row['mean_fwd_ic']:+.4f}  "
              f"beats_p95={row['n_beats_p95']}/{row['n_assets']}")

    outp = ROOT / "runs" / "research" / "ma_dna_cadence_scan_result.json"
    outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {outp}")
