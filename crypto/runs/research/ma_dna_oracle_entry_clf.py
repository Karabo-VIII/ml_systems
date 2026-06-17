"""runs/research/ma_dna_oracle_entry_clf.py

TASK (oracle-decomposition step 2, per-CELL): fit P(oracle-entry | causal MA-DNA features) as a
HELD-OUT classifier -- fit on TRAIN+VAL, SCORE ON UNSEEN ONLY (the model never sees UNSEEN) -- and
report, PER CELL (= per (asset, cadence)):

  * AUC      : ROC-AUC of the DNA score vs the oracle-ENTRY label on UNSEEN  (classification discrimination)
  * fwd_IC   : Spearman IC of the DNA score vs the realized forward open->open return on UNSEEN
               (the tradeable-skill honesty metric: classification lift != forward edge)
  * label_IC : Spearman IC of the DNA score vs the oracle-ENTRY label on UNSEEN
               (the literal "IC of the DNA score against the oracle-entry label"; for a binary label this
                is monotonically tied to AUC -- reported for completeness)
  * a shuffled-label NULL AUC (permute FIT labels, refit, N seeds) so the AUC has a calibrated baseline.

DNA FEATURE BASIS = the task's named causal MA features, all strictly past-only (close.shift(1) base):
  1/2/3-MA distance + slope + gap + cross + r(efficiency-ratio)
  -> reuses the audited build_ma_groups(): single (dist20/slope20/dist50) + two_ma (gap_10_30/
     crossstate_10_30/gap_20_50) + ribbon3 (order_score/compression/ribbon_dist) + kama_er (er10/er20/
     kama_dist). The double-smoothed ma_of_ma group is NOT in the task's named list, so it is EXCLUDED
     from the primary DNA set; an "ALL" variant (DNA + ma_of_ma) is reported as a secondary column.
  Cross enters as ONE regularized feature (cross-STATE sign), NOT a standalone MA-cross trigger (REFUTED).

MODEL: L2 logistic (C=0.5, class_weight=balanced), StandardScaler fit on TRAIN+VAL only.

REUSES audited primitives (does NOT reinvent): oracle_high_capture + WIN (oracle_ceiling_builder.py,
DP self-tested), build_ma_groups + fwd_open_to_open + load_ohlc (ma_dna_cadence_scan.py, past-only
falsifier in its selftest).

HARD CONSTRAINTS: LONG-ONLY, past-only features, fit TRAIN+VAL, UNSEEN scored once, taker 0.0024.

RWYB:
    .venv/Scripts/python.exe runs/research/ma_dna_oracle_entry_clf.py --selftest
    .venv/Scripts/python.exe runs/research/ma_dna_oracle_entry_clf.py            # full grid
    .venv/Scripts/python.exe runs/research/ma_dna_oracle_entry_clf.py --quick    # BTC/ETH/SOL x 4h,1d
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

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# audited, self-tested primitives -- reused verbatim, not reinvented
from ma_dna_cadence_scan import build_ma_groups, fwd_open_to_open, load_ohlc  # noqa: E402
from oracle_ceiling_builder import oracle_high_capture, WIN  # noqa: E402

COST_RT = 0.0024

# the task's named causal MA features map exactly onto these four audited groups:
#   1/2/3-MA distance + slope -> single ; gap + cross -> two_ma ; 3-MA ribbon -> ribbon3 ; r -> kama_er
DNA_GROUPS = ["single", "two_ma", "ribbon3", "kama_er"]
ALL_GROUPS = ["single", "two_ma", "ribbon3", "kama_er", "ma_of_ma"]  # secondary (adds double-smoothed)

__contract__ = {
    "kind": "ma_dna_oracle_entry_held_out_classifier",
    "version": "1.0",
    "inputs": ["chimera (asset,cadence) ohlc+ts", "oracle-entry labels from oracle_high_capture DP"],
    "outputs": ["per-cell: UNSEEN AUC(vs label), fwd_IC(vs fwd ret), label_IC, shuffled-null AUC, lift"],
    "invariants": [
        "DNA features strictly past-only (build_ma_groups close.shift(1)); NO ohlc/target/fwd cols in X",
        "fit on TRAIN+VAL only; SCORE on UNSEEN only (OOS excluded from scoring); shuffle permutes FIT labels",
        "DNA basis = single+two_ma+ribbon3+kama_er (task's named feats); ma_of_ma only in the ALL variant",
        "cross-STATE is one regularized feature, never a standalone MA-cross trigger (REFUTED family)",
        "AUC = classification discrimination of oracle entries; fwd_IC = forward-return skill (kept separate)",
    ],
}


def _wmask(ts_ms: np.ndarray, name: str) -> np.ndarray:
    lo, hi = WIN[name]
    lo_ms = 0 if lo == "0" else int(pd.Timestamp(lo).value // 1_000_000)
    hi_ms = int(pd.Timestamp(hi).value // 1_000_000)
    return (ts_ms >= lo_ms) & (ts_ms < hi_ms)


def _dna_matrix(cl: np.ndarray, groups: list[str]) -> tuple[list[str], np.ndarray]:
    """Concatenate the named past-only MA groups into one feature matrix."""
    g = build_ma_groups(cl)
    names, cols = [], []
    for gname in groups:
        gn, gc = g[gname]
        names += [f"{gname}.{k}" for k in gn]
        cols.append(gc)
    return names, np.column_stack(cols)


def _fit_score_unseen(X, y, fwd, m_fit, m_unseen, n_shuffle, seed=7):
    """Fit L2 logistic on TRAIN+VAL; score UNSEEN. Return per-cell metrics or None if not evaluable."""
    valid = np.all(np.isfinite(X), axis=1)
    fit_idx = np.where(m_fit & valid)[0]
    un_idx = np.where(m_unseen & valid)[0]
    if len(fit_idx) < 200 or len(un_idx) < 80:
        return None
    Xtr, ytr = X[fit_idx], y[fit_idx]
    Xun, yun = X[un_idx], y[un_idx]
    fun = fwd[un_idx]
    if len(np.unique(ytr)) < 2 or len(np.unique(yun)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xun_s = sc.transform(Xtr), sc.transform(Xun)

    def _fit_predict(yfit):
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr_s, yfit)
        return clf.predict_proba(Xun_s)[:, 1]

    p = _fit_predict(ytr)
    auc = float(roc_auc_score(yun, p))
    label_ic = float(spearmanr(p, yun)[0])
    mm = np.isfinite(fun) & np.isfinite(p)
    fwd_ic = float(spearmanr(p[mm], fun[mm])[0]) if mm.sum() > 20 else float("nan")

    sh = []
    for s in range(n_shuffle):
        rng = np.random.default_rng(1000 + s)
        ys = ytr[rng.permutation(len(ytr))]
        if len(np.unique(ys)) < 2:
            continue
        sh.append(float(roc_auc_score(yun, _fit_predict(ys))))
    sh = np.asarray(sh, float)
    sh_mean = float(sh.mean()) if len(sh) else float("nan")
    sh_p95 = float(np.percentile(sh, 95)) if len(sh) else float("nan")
    return {
        "auc_unseen": round(auc, 4),
        "fwd_ic_unseen": round(fwd_ic, 4) if np.isfinite(fwd_ic) else None,
        "label_ic_unseen": round(label_ic, 4),
        "shuffled_mean_auc": round(sh_mean, 4),
        "shuffled_p95_auc": round(sh_p95, 4),
        "auc_lift": round(auc - sh_mean, 4),
        "beats_shuffled_p95": bool(auc > sh_p95),
        "n_fit": int(len(fit_idx)),
        "n_unseen": int(len(un_idx)),
        "unseen_entries": int(yun.sum()),
        "unseen_base_rate": round(float(yun.mean()), 4),
        "fit_base_rate": round(float(ytr.mean()), 4),
    }


def run_cell(asset, cadence, n_shuffle, with_all=True, verbose=True):
    ts, op, hi, cl = load_ohlc(asset, cadence)
    n = len(op)
    f_dp, trades = oracle_high_capture(ts, op, hi)
    if not trades:
        return {"asset": asset, "cadence": cadence, "error": "no oracle trades"}
    y = np.zeros(n, dtype=int)
    ent = np.array([i for i, j in trades], dtype=int)
    y[ent] = 1
    holds = np.array([j - i for i, j in trades], dtype=int)
    H = int(max(1, np.median(holds)))            # forward horizon = oracle median hold (the move length)
    fwd = fwd_open_to_open(op, H)

    m_fit = _wmask(ts, "TRAIN") | _wmask(ts, "VAL")
    m_unseen = _wmask(ts, "UNSEEN")

    out = {"asset": asset, "cadence": cadence, "n_bars": int(n),
           "oracle_entries": int(y.sum()), "oracle_base_rate": round(float(y.mean()), 4),
           "exit_H_bars": int(H)}

    names, X = _dna_matrix(cl, DNA_GROUPS)
    out["dna_features"] = names
    out["DNA"] = _fit_score_unseen(X, y, fwd, m_fit, m_unseen, n_shuffle)
    if with_all:
        _, Xa = _dna_matrix(cl, ALL_GROUPS)
        out["ALL"] = _fit_score_unseen(Xa, y, fwd, m_fit, m_unseen, n_shuffle)

    if verbose:
        d = out["DNA"]
        if d is None:
            print(f"  {asset:6} {cadence:5}  --not evaluable (insufficient UNSEEN/class)--")
        else:
            star = "*" if d["beats_shuffled_p95"] else " "
            print(f" {star}{asset:6} {cadence:5}  AUC={d['auc_unseen']:.3f} "
                  f"shuf(m/p95)={d['shuffled_mean_auc']:.3f}/{d['shuffled_p95_auc']:.3f} "
                  f"lift={d['auc_lift']:+.3f}  fwd_IC={d['fwd_ic_unseen']}  "
                  f"label_IC={d['label_ic_unseen']:+.3f}  "
                  f"n_un={d['n_unseen']}(ent {d['unseen_entries']}) H={H}")
    return out


def run_grid(cells, n_shuffle):
    results = []
    print(f"\n{'='*92}\nMA-DNA -> P(oracle-entry) HELD-OUT (fit TRAIN+VAL, score UNSEEN)  n_shuffle={n_shuffle}")
    print(f"DNA = 1/2/3-MA distance + slope + gap + cross + r  (single+two_ma+ribbon3+kama_er)\n{'='*92}")
    for asset, cad in cells:
        try:
            results.append(run_cell(asset, cad, n_shuffle))
        except Exception as e:
            print(f"  {asset:6} {cad:5}  ERROR {type(e).__name__}: {str(e)[:70]}")
            results.append({"asset": asset, "cadence": cad, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    # ---- aggregate per cadence (mean over evaluable cells) ----
    agg = {}
    for r in results:
        d = r.get("DNA")
        if not d:
            continue
        c = r["cadence"]
        agg.setdefault(c, {"auc": [], "lift": [], "fwd": [], "lab": [], "beats": 0, "n": 0})
        agg[c]["auc"].append(d["auc_unseen"])
        agg[c]["lift"].append(d["auc_lift"])
        if d["fwd_ic_unseen"] is not None:
            agg[c]["fwd"].append(d["fwd_ic_unseen"])
        agg[c]["lab"].append(d["label_ic_unseen"])
        agg[c]["beats"] += int(d["beats_shuffled_p95"])
        agg[c]["n"] += 1
    per_cadence = []
    for c, d in agg.items():
        per_cadence.append({
            "cadence": c, "n_cells": d["n"],
            "mean_auc": round(float(np.mean(d["auc"])), 4),
            "mean_lift": round(float(np.mean(d["lift"])), 4),
            "mean_fwd_ic": round(float(np.mean(d["fwd"])), 4) if d["fwd"] else None,
            "mean_label_ic": round(float(np.mean(d["lab"])), 4),
            "n_beats_p95": d["beats"],
        })
    per_cadence.sort(key=lambda x: x["mean_lift"], reverse=True)
    return {"per_cell": results, "per_cadence": per_cadence,
            "window_def": WIN, "dna_groups": DNA_GROUPS, "cost_rt": COST_RT}


# ===========================================================================
def _selftest():
    print("=" * 70)
    print("[ma_dna_oracle_entry_clf selftest]")
    print("=" * 70)
    ok = True
    rng = np.random.default_rng(0)
    n = 8000
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    # (1) past-only falsifier: bumping close[i] must NOT change DNA feature row i (only row i+1)
    names, X = _dna_matrix(close, DNA_GROUPS)
    c2 = close.copy(); c2[4000] *= 1.4
    _, X2 = _dna_matrix(c2, DNA_GROUPS)
    same_i = np.allclose(np.nan_to_num(X[4000]), np.nan_to_num(X2[4000]))
    chg_next = not np.allclose(np.nan_to_num(X[4001]), np.nan_to_num(X2[4001]))
    print(f"  past-only: row4000 unchanged={same_i} (expect True); row4001 changed={chg_next} (expect True)")
    ok &= same_i and chg_next
    # (2) power: a label built from a real past-only DNA feature -> AUC>0.6, shuffled-null ~0.5
    feat = X[:, 0]
    valid = np.isfinite(feat)
    y = np.zeros(n, int)
    thr = np.nanmedian(feat)
    y[valid] = (feat[valid] + 0.25 * rng.normal(0, np.nanstd(feat[valid]), valid.sum()) > thr).astype(int)
    fwd = rng.normal(0, 0.02, n)
    m_fit = np.zeros(n, bool); m_fit[:5500] = True
    m_un = np.zeros(n, bool); m_un[5500:] = True
    r = _fit_score_unseen(X, y, fwd, m_fit, m_un, n_shuffle=20)
    print(f"  power: AUC={r['auc_unseen']:.3f}(expect>0.6) shuf_mean={r['shuffled_mean_auc']:.3f}(expect~0.5) "
          f"beats_p95={r['beats_shuffled_p95']}")
    ok &= r["auc_unseen"] > 0.6 and abs(r["shuffled_mean_auc"] - 0.5) < 0.07 and r["beats_shuffled_p95"]
    # (3) soundness floor: a label INDEPENDENT of features -> AUC ~0.5, should NOT beat shuffled p95
    y_rand = (rng.random(n) < 0.2).astype(int)
    r2 = _fit_score_unseen(X, y_rand, fwd, m_fit, m_un, n_shuffle=20)
    print(f"  null-label: AUC={r2['auc_unseen']:.3f}(expect~0.5) beats_p95={r2['beats_shuffled_p95']}(expect False)")
    ok &= abs(r2["auc_unseen"] - 0.5) < 0.08 and not r2["beats_shuffled_p95"]
    print(f"\n[selftest] {'PASS' if ok else 'FAIL'} -- gate is past-only, has power, and rejects a null label.")
    return ok


DEFAULT_MAIN = ["BTC", "ETH", "SOL", "BNB", "AVAX", "ADA", "DOGE", "LINK", "XRP", "PEPE"]
DEFAULT_EVENT = ["BTC", "ETH", "PEPE"]


def build_cells(args):
    main_assets = args.assets.split(",") if args.assets else DEFAULT_MAIN
    cells = []
    for cad in args.cadences.split(","):
        pool = DEFAULT_EVENT if cad in ("range", "dib") else main_assets
        for a in pool:
            cells.append((a, cad))
    return cells


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="")
    ap.add_argument("--cadences", default="4h,1d,1h,15m,range,dib")
    ap.add_argument("--n-shuffle", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not _selftest():
        print("SELFTEST FAILED -- aborting before touching market data")
        sys.exit(1)
    if args.quick:
        args.assets = "BTC,ETH,SOL"
        args.cadences = "4h,1d"
        args.n_shuffle = 12
    cells = build_cells(args)
    res = run_grid(cells, args.n_shuffle)

    print("\n" + "=" * 92)
    print("PER-CADENCE SUMMARY (mean over evaluable cells), ranked by AUC-lift over shuffled null")
    print("=" * 92)
    print(f"  {'cadence':8} {'n':>3}  {'mean_AUC':>8} {'mean_lift':>9} {'beats_p95':>10} "
          f"{'mean_fwdIC':>11} {'mean_labIC':>11}")
    for row in res["per_cadence"]:
        print(f"  {row['cadence']:8} {row['n_cells']:>3}  {row['mean_auc']:>8.4f} {row['mean_lift']:>+9.4f} "
              f"{str(row['n_beats_p95'])+'/'+str(row['n_cells']):>10} "
              f"{(row['mean_fwd_ic'] if row['mean_fwd_ic'] is not None else float('nan')):>+11.4f} "
              f"{row['mean_label_ic']:>+11.4f}")

    outp = ROOT / "runs" / "research" / "ma_dna_oracle_entry_clf_result.json"
    outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {outp}")
