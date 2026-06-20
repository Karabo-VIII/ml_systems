"""src/strat/capture_sweep.py -- MOVE-CATCH PRODUCT SWEEP (2026-06-20).

Sweep: price TIs x TF {1d,4h} x exit {time,trail5,target8} x by_regime=True.
Ranking metric: chop+bear realized_net (NOT bull -- bull = beta).
Adversarial battery: Holm/BH-correct p across the whole sweep.
Also: hold sweep + min_move sweep for top non-bull TIs.

DEV wall (<= 2024-05-15). Long-only spot, taker cost, causal. No emoji.
RWYB: python -m strat.capture_sweep [--quick]
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
import strat.capture_lab as cl

PRICE_TIS = ["mom7", "mom14", "mom30", "rsi14", "brk14", "rangepos", "volexp", "accel"]
TFS = ["1d", "4h"]
EXITS = ["time", "trail5", "target8"]
HOLDS_1D = [5, 7, 10, 14]    # days (= bars for 1d)
HOLDS_4H = [6, 12, 18, 28]   # bars (~1d, 2d, 3d, 4.5d)
MIN_MOVES = [0.02, 0.03, 0.05]


def holm_correct(pvals):
    """Holm-Bonferroni stepwise correction. Returns adjusted p-values (same length as input)."""
    n = len(pvals)
    if n == 0: return np.array([])
    idx = np.argsort(pvals)
    adj = np.zeros(n)
    for rank, i in enumerate(idx):
        adj[i] = min(1.0, pvals[i] * (n - rank))
    # monotone: each adjusted p >= previous (in sorted order)
    for k in range(1, n):
        adj[idx[k]] = max(adj[idx[k]], adj[idx[k-1]])
    return adj


def bh_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean reject array + adjusted p-values."""
    n = len(pvals)
    if n == 0: return np.zeros(n, bool), np.ones(n)
    idx = np.argsort(pvals)
    threshold = (np.arange(1, n+1) / n) * alpha
    sorted_p = np.array(pvals)[idx]
    reject_sorted = sorted_p <= threshold
    # find last True; everything up to it rejects
    last = -1
    for k in range(n-1, -1, -1):
        if reject_sorted[k]: last = k; break
    rej = np.zeros(n, bool)
    if last >= 0:
        rej[idx[:last+1]] = True
    adj = np.zeros(n)
    for rank, i in enumerate(idx):
        adj[i] = min(1.0, pvals[i] * n / (rank+1))
    return rej, adj


def run_main_sweep(labs, quick=False):
    """TI x TF x exit product sweep with by_regime + Holm/BH correction."""
    print("\n=== MAIN SWEEP: TI x TF x exit x regime ===")
    records = []
    n_null = 200 if quick else 400
    combos = [(ti, tf, ex) for ti in PRICE_TIS for tf in TFS for ex in EXITS]
    total = len(combos)
    for k, (ti, tf, ex) in enumerate(combos):
        lab = labs[tf]
        hold = 7 * fl.BARS_PER_DAY[tf]
        r = cl.evaluate_ti(lab, ti, tf=tf, hold=hold, exit_kind=ex, min_move=0.03,
                           n_null=n_null, by_regime=True)
        if "note" in r:
            print(f"  [{k+1}/{total}] {ti:10} {tf:4} {ex:9} SKIP ({r['note']})")
            continue
        rec = {"ti": ti, "tf": tf, "exit": ex,
               "n_fired": r["n_fired"],
               "mean_MFE_fired": r["mean_MFE_fired"],
               "mean_realized_net": r["mean_realized_net"],
               "null_realized_net": r["null_realized_net"],
               "edge_vs_random_pp": r["edge_vs_random_pp"],
               "p_vs_random": r["p_vs_random"],
               "capture_rate": r["capture_rate"]}
        # regime breakdown
        for rg in ("bull", "chop", "bear"):
            d = r.get("by_regime", {}).get(rg)
            if d:
                rec[f"{rg}_real"] = d["realized_net"]
                rec[f"{rg}_null"] = d["null_net"]
                rec[f"{rg}_edge"] = d["edge_pp"]
                rec[f"{rg}_p"] = d["p_vs_random"]
                rec[f"{rg}_n"] = d["n"]
            else:
                rec[f"{rg}_real"] = np.nan; rec[f"{rg}_null"] = np.nan
                rec[f"{rg}_edge"] = np.nan; rec[f"{rg}_p"] = np.nan; rec[f"{rg}_n"] = 0
        records.append(rec)
        print(f"  [{k+1}/{total}] {ti:10} {tf:4} {ex:9} "
              f"n={r['n_fired']:>5} edge={r['edge_vs_random_pp']:>5.2f}pp p={r['p_vs_random']:.3f} "
              f"| bull_e={rec['bull_edge']:>5.2f} chop_e={rec['chop_edge']:>5.2f} bear_e={rec['bear_edge']:>5.2f}")

    df = pd.DataFrame(records)
    if df.empty:
        print("  NO RECORDS -- all combos skipped.")
        return df

    # Holm/BH across all regime-level p-values in the sweep
    # collect (chop_p, bear_p) raw values for correction
    all_ps_chop = df["chop_p"].dropna().tolist()
    all_ps_bear = df["bear_p"].dropna().tolist()
    all_ps = all_ps_chop + all_ps_bear
    if all_ps:
        holm = holm_correct(all_ps)
        rej_bh, bh_adj = bh_correct(all_ps)
        nc = len(all_ps_chop)
        df_chop_idx = df[df["chop_p"].notna()].index
        df_bear_idx = df[df["bear_p"].notna()].index
        df.loc[df_chop_idx, "chop_p_holm"] = holm[:nc]
        df.loc[df_bear_idx, "bear_p_holm"] = holm[nc:]
        df.loc[df_chop_idx, "chop_bh_rej"] = rej_bh[:nc]
        df.loc[df_bear_idx, "bear_bh_rej"] = rej_bh[nc:]

    # ranking metric: mean(chop_edge, bear_edge) -- non-bull composite
    df["nonbull_edge"] = df[["chop_edge", "bear_edge"]].mean(axis=1)
    df = df.sort_values("nonbull_edge", ascending=False)

    print("\n--- TOP NON-BULL COMBOS (ranked by mean chop+bear edge vs random) ---")
    cols = ["ti","tf","exit","n_fired","chop_edge","chop_p","chop_p_holm","bear_edge","bear_p","bear_p_holm","nonbull_edge","capture_rate"]
    display_cols = [c for c in cols if c in df.columns]
    print(df[display_cols].head(20).to_string(index=False))

    print("\n--- BH-SURVIVORS (non-bull regime, FDR 5%) ---")
    bh_chop = df[df.get("chop_bh_rej", pd.Series(False, index=df.index)).fillna(False)]
    bh_bear = df[df.get("bear_bh_rej", pd.Series(False, index=df.index)).fillna(False)]
    if len(bh_chop):
        print(f"  Chop BH survivors ({len(bh_chop)}):")
        print(bh_chop[["ti","tf","exit","chop_edge","chop_p","capture_rate"]].to_string(index=False))
    else:
        print("  Chop: NO BH survivors")
    if len(bh_bear):
        print(f"  Bear BH survivors ({len(bh_bear)}):")
        print(bh_bear[["ti","tf","exit","bear_edge","bear_p","capture_rate"]].to_string(index=False))
    else:
        print("  Bear: NO BH survivors")

    return df


def run_hold_sweep(labs, top_tis, quick=False):
    """Hold sweep for top non-bull TIs: does a different hold window sharpen the edge?"""
    print("\n=== HOLD SWEEP (top non-bull TIs, 1d + 4h) ===")
    n_null = 150 if quick else 300
    records = []
    for tf in TFS:
        lab = labs[tf]; bpd = fl.BARS_PER_DAY[tf]
        holds = HOLDS_1D if tf == "1d" else HOLDS_4H
        for ti in top_tis:
            for h in holds:
                r = cl.evaluate_ti(lab, ti, tf=tf, hold=h, exit_kind="time", min_move=0.03,
                                   n_null=n_null, by_regime=True)
                if "note" in r: continue
                rec = {"ti": ti, "tf": tf, "hold": h, "hold_days": round(h / bpd, 1),
                       "edge_vs_random_pp": r["edge_vs_random_pp"], "p_vs_random": r["p_vs_random"]}
                for rg in ("chop", "bear"):
                    d = r.get("by_regime", {}).get(rg)
                    rec[f"{rg}_edge"] = d["edge_pp"] if d else np.nan
                    rec[f"{rg}_p"] = d["p_vs_random"] if d else np.nan
                records.append(rec)
                print(f"  {ti:10} {tf:4} hold={h:>3}bars ({rec['hold_days']:>4}d)  "
                      f"chop_e={rec['chop_edge']:>5.2f} bear_e={rec['bear_edge']:>5.2f}")
    df = pd.DataFrame(records)
    if not df.empty:
        df["nonbull"] = df[["chop_edge","bear_edge"]].mean(axis=1)
        print("\n  Top hold configs (non-bull):")
        print(df.sort_values("nonbull", ascending=False).head(15).to_string(index=False))
    return df


def run_minmove_sweep(labs, top_tis, quick=False):
    """min_move threshold sweep: does filtering to bigger available moves sharpen the edge?"""
    print("\n=== MIN_MOVE SWEEP (top non-bull TIs, 1d) ===")
    n_null = 150 if quick else 300
    records = []
    lab = labs["1d"]
    for ti in top_tis:
        for mm in MIN_MOVES:
            r = cl.evaluate_ti(lab, ti, tf="1d", hold=7, exit_kind="time", min_move=mm,
                               n_null=n_null, by_regime=True)
            if "note" in r: continue
            rec = {"ti": ti, "min_move": mm, "n_fired": r["n_fired"],
                   "edge_vs_random_pp": r["edge_vs_random_pp"], "p_vs_random": r["p_vs_random"]}
            for rg in ("chop", "bear"):
                d = r.get("by_regime", {}).get(rg)
                rec[f"{rg}_edge"] = d["edge_pp"] if d else np.nan
                rec[f"{rg}_p"] = d["p_vs_random"] if d else np.nan
            records.append(rec)
            print(f"  {ti:10} min_move={mm:.2f}  n={r['n_fired']:>5}  "
                  f"chop_e={rec['chop_edge']:>5.2f} bear_e={rec['bear_edge']:>5.2f}  p={r['p_vs_random']:.3f}")
    df = pd.DataFrame(records)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer nulls, faster run")
    ap.add_argument("--out", default="", help="optional JSON output path")
    a = ap.parse_args()
    quick = a.quick

    print(f"[capture_sweep] DEV wall <= {fl.DEV_END}  quick={quick}")
    print("Loading DEV labs (1d + 4h)...")
    labs = {}
    for tf in TFS:
        bpd = fl.BARS_PER_DAY[tf]
        lab = fl.load_wide(n=50, tf=tf, min_bars=(200 * bpd if tf != "1d" else 400))
        C = lab["C"]
        assert C.index.max() < pd.Timestamp(fl.DEV_END), f"WALL VIOLATION {tf}"
        print(f"  {tf}: {len(lab['syms'])} assets {C.index.min().date()} -> {C.index.max().date()}")
        labs[tf] = lab

    # ---- 1. Main product sweep ----
    df_main = run_main_sweep(labs, quick=quick)

    # ---- 2. Identify top non-bull TIs for hold+minmove sweeps ----
    top_tis = ["mom14", "brk14"]   # defaults
    if not df_main.empty and "nonbull_edge" in df_main.columns:
        by_ti = df_main.groupby("ti")["nonbull_edge"].mean().sort_values(ascending=False)
        top_tis = by_ti.head(4).index.tolist()
        print(f"\n  Top non-bull TIs from main sweep: {top_tis}")

    # ---- 3. Hold sweep ----
    df_hold = run_hold_sweep(labs, top_tis, quick=quick)

    # ---- 4. Min-move sweep ----
    df_mm = run_minmove_sweep(labs, top_tis, quick=quick)

    # ---- 5. Summary + verdict ----
    print("\n=== FINAL VERDICT ===")
    if not df_main.empty:
        df_chop_pos = df_main[(df_main["chop_edge"] > 0) & (df_main["chop_p"] < 0.05)]
        df_bear_pos = df_main[(df_main["bear_edge"] > 0) & (df_main["bear_p"] < 0.05)]
        bh_col_c = "chop_bh_rej" if "chop_bh_rej" in df_main.columns else None
        bh_col_b = "bear_bh_rej" if "bear_bh_rej" in df_main.columns else None
        n_chop_raw = len(df_chop_pos)
        n_bear_raw = len(df_bear_pos)
        n_chop_bh = int(df_main[bh_col_c].fillna(False).sum()) if bh_col_c else 0
        n_bear_bh  = int(df_main[bh_col_b].fillna(False).sum()) if bh_col_b else 0
        print(f"  CHOP: {n_chop_raw} combos p<0.05 (raw), {n_chop_bh} survive BH-FDR-5%")
        print(f"  BEAR: {n_bear_raw} combos p<0.05 (raw), {n_bear_bh} survive BH-FDR-5%")
        print(f"  Total combos evaluated: {len(df_main)}")
        print(f"  NOTE: p-values are based on permutation resampling; IID assumption may overstate significance.")
        print(f"        A date-block bootstrap (honest N_eff) is the next required validation step.")

    # ---- 6. Save JSON ----
    out_path = a.out
    if not out_path:
        runs_dir = Path(__file__).resolve().parents[2] / "runs" / "strat"
        runs_dir.mkdir(parents=True, exist_ok=True)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(runs_dir / f"capture_sweep_{ts}.json")
    results = {
        "main_sweep": df_main.to_dict(orient="records") if not df_main.empty else [],
        "hold_sweep": df_hold.to_dict(orient="records") if not df_hold.empty else [],
        "minmove_sweep": df_mm.to_dict(orient="records") if not df_mm.empty else [],
    }
    with open(out_path, "w") as fh:
        json.dump(results, fh, default=str, indent=2)
    print(f"\n  Results saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
