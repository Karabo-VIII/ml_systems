"""DEAD/NULL CASE REWORK — interrogate the two most informative failures before
accepting them as dead. Uses the AUDITED harness from ti_dna_report.

Case 1 — ADX (the ONLY genuinely selective family, ~70% deploy, but VAL/OOS-INCONSISTENT):
  Selectivity is the rare/valuable property. If a lookback x threshold setting makes ADX
  BOTH selective AND VAL/OOS-consistent + beats null, that would be the first true selective
  engine. Sweep lookback {7,10,14,21,28} x threshold {0.25..0.6} at 1d/4h.

Case 2 — STOCHASTIC (worst bleeder: -7.8pp/wk, negative Sharpe). A signal that loses that
  consistently may be INVERTED (picking the worst entries). Test: does flipping the signal
  (1-held) turn the bleed into an edge? If yes -> the logic is backwards (a real finding /
  fix). If no (both sides bleed) -> genuinely no signal (confirmed dead).

Rule: a dead case is only confirmed dead AFTER it survives this interrogation.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ti_engine_sota import adx_single_held, stochastic_single_held, STOCHASTIC_CONFIGS
from strat.ti_dna_report import (
    btc_regime, replay, profile, random_null,
    DEV_WALL_MS, OOS_WALL_MS, VAL_START, VAL_END, OOS_START, OOS_2024_END,
)

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V5_dna_report"


def _panel_from_holds(hold_fn, tf, wall_ms):
    """Build close/held panels using a custom per-asset hold function hold_fn(o,h,l,c)."""
    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        held = hold_fn(o, h, l, np.asarray(c, float))
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)
    if len(close_cols) < 10:
        return None, None
    cdf = pd.DataFrame(close_cols).sort_index()
    hdf = pd.DataFrame(held_cols).reindex(cdf.index).fillna(0).astype(np.int8)
    return cdf, hdf


def eval_cell(hold_fn, tf, n_null=100):
    """Return (VAL profile, VAL null, OOS profile, OOS null) for a custom hold fn."""
    bpd = BARS_PER_DAY[tf]
    reg_dev = btc_regime(DEV_WALL_MS); reg_oos = btc_regime(OOS_WALL_MS)
    cdf, hdf = _panel_from_holds(hold_fn, tf, DEV_WALL_MS)
    if cdf is None:
        return None
    pv = replay(cdf, hdf, reg_dev, VAL_START, VAL_END); mv = profile(pv, bpd)
    vn = float(np.mean(random_null(pv, True, n_null, pv["deploy"]) < mv["compound"])) * 100 if pv["deploy"] > 1e-3 else 0
    cdfo, hdfo = _panel_from_holds(hold_fn, tf, OOS_WALL_MS)
    po = replay(cdfo, hdfo, reg_oos, OOS_START, OOS_2024_END); mo = profile(po, bpd)
    on = float(np.mean(random_null(po, True, n_null, po["deploy"]) < mo["compound"])) * 100 if po["deploy"] > 1e-3 else 0
    return mv, vn, mo, on


# --------------------------------------------------------------------------- #
def adx_ensemble_hold(lookback, threshold):
    """Single-param ADX (efficiency-ratio) ensemble of one — sweep direct."""
    def fn(o, h, l, c):
        return adx_single_held(c, lookback, threshold)
    return fn


def rework_adx(n_null=100):
    print("\n" + "=" * 96)
    print("CASE 1 — ADX REWORK (only selective family; can it become selective AND consistent?)")
    print("=" * 96)
    print(f"  {'TF':<4} {'lb':>3} {'thr':>5} | {'VALedge':>8} {'VALnull':>7} {'VALdep':>6} | "
          f"{'OOSedge':>8} {'OOSnull':>7} {'OOSdep':>6} | verdict")
    print("  " + "-" * 88)
    rows = []
    for tf in ["1d", "4h"]:
        for lb in [7, 10, 14, 21, 28]:
            for thr in [0.25, 0.35, 0.45, 0.55]:
                res = eval_cell(adx_ensemble_hold(lb, thr), tf, n_null)
                if res is None:
                    continue
                mv, vn, mo, on = res
                sel = mv["deploy"] < 0.70 and mo["deploy"] < 0.70
                cons = mv["edge_pp"] > 0 and mo["edge_pp"] > 0
                nullok = on >= 95
                if sel and cons and nullok:
                    verdict = "*** SELECTIVE+ROBUST"
                elif sel and cons:
                    verdict = "selective+consistent"
                elif cons and nullok:
                    verdict = "consistent (not sel)"
                elif sel:
                    verdict = "selective (not cons)"
                else:
                    verdict = "null"
                rows.append({"tf": tf, "lookback": lb, "threshold": thr,
                             "val_edge": mv["edge_pp"], "val_null": round(vn, 0), "val_deploy": mv["deploy"],
                             "oos_edge": mo["edge_pp"], "oos_null": round(on, 0), "oos_deploy": mo["deploy"],
                             "verdict": verdict})
                print(f"  {tf:<4} {lb:>3} {thr:>5.2f} | {mv['edge_pp']:>+7.3f} {vn:>6.0f} {mv['deploy']:>5.1%} | "
                      f"{mo['edge_pp']:>+7.3f} {on:>6.0f} {mo['deploy']:>5.1%} | {verdict}")
    return rows


def rework_stochastic(n_null=100):
    print("\n" + "=" * 96)
    print("CASE 2 — STOCHASTIC INVERSION (is the bleed a backwards-logic bug or genuine null?)")
    print("=" * 96)
    cfgs = STOCHASTIC_CONFIGS

    def stoch_normal(o, h, l, c):
        n = len(c); votes = np.zeros(n)
        for cfg in cfgs:
            votes += stochastic_single_held(c, h, l, *cfg).astype(float)
        return (votes / len(cfgs) > 0.5).astype(np.int8)

    def stoch_inverted(o, h, l, c):
        return (1 - stoch_normal(o, h, l, c)).astype(np.int8)

    print(f"  {'variant':<10} {'TF':<4} | {'VALedge':>8} {'VALdep':>6} | {'OOSedge':>8} {'OOSdep':>6} | read")
    print("  " + "-" * 76)
    rows = []
    for tf in ["1d", "4h", "2h"]:
        for name, fn in [("normal", stoch_normal), ("inverted", stoch_inverted)]:
            res = eval_cell(fn, tf, n_null)
            if res is None:
                continue
            mv, vn, mo, on = res
            rows.append({"variant": name, "tf": tf, "val_edge": mv["edge_pp"], "val_deploy": mv["deploy"],
                         "oos_edge": mo["edge_pp"], "oos_deploy": mo["deploy"]})
            print(f"  {name:<10} {tf:<4} | {mv['edge_pp']:>+7.3f} {mv['deploy']:>5.1%} | "
                  f"{mo['edge_pp']:>+7.3f} {mo['deploy']:>5.1%} |")
    # interpret
    norm = {r["tf"]: r for r in rows if r["variant"] == "normal"}
    inv = {r["tf"]: r for r in rows if r["variant"] == "inverted"}
    print("\n  Interpretation:")
    for tf in ["1d", "4h", "2h"]:
        if tf in norm and tf in inv:
            n_oos, i_oos = norm[tf]["oos_edge"], inv[tf]["oos_edge"]
            if i_oos > 0.3 and n_oos < -0.3:
                print(f"    {tf}: INVERTED beats (+{i_oos:.2f}) while normal bleeds ({n_oos:.2f}) "
                      f"-> logic may be BACKWARDS (finding).")
            elif n_oos < 0 and i_oos < 0:
                print(f"    {tf}: BOTH sides lose (norm {n_oos:.2f}, inv {i_oos:.2f}) "
                      f"-> genuinely NO signal (confirmed dead).")
            else:
                print(f"    {tf}: norm {n_oos:.2f}, inv {i_oos:.2f} -> mixed.")
    return rows


def main():
    t0 = time.time()
    adx_rows = rework_adx()
    stoch_rows = rework_stochastic()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"deadcase_rework_{ts}.json", "w") as f:
        json.dump({"adx": adx_rows, "stochastic": stoch_rows, "timestamp": ts}, f, indent=2, default=str)
    print(f"\n  Done in {time.time()-t0:.0f}s. Saved deadcase_rework_{ts}.json")


if __name__ == "__main__":
    main()
