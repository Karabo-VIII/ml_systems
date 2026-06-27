"""ADVERSARIAL AUDIT of the ADX|1d SELECTIVE (PROVISIONAL) claim (Claude, 2026-06-28).

Presume broken until proven. Re-run the ADX|1d lb22-28 x thr0.42-0.48 sweep on the EXACT
verified harness (ti_dna_report.replay/profile/random_null + adx_single_held), and:

  1. Reproduce VAL + OOS edge, deploy, 100th-pct null vs the claimed numbers.
  2. Apply the SAME de-leak the leakcheck mandated (regime.shift(1)) -- the replay() regime
     broadcast (lines 192-197) is the identical leaked same-bar gate. Does ADX survive de-leak?
  3. Quantify VAL/OOS shrinkage (is most of the edge in-sample?).
  4. Directional-gradient check: edge vs lookback (claim: lb>=22 consistently +, <20 dies).
  5. Random-null reproduction at matched deploy.

de-leak: pass a pre-shifted regime to replay so day-D reads regime[D-1] (last CLOSED 1d bar).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ti_engine_sota import adx_single_held
from strat.ti_dna_report import (
    btc_regime, replay, profile, random_null,
    DEV_WALL_MS, OOS_WALL_MS, VAL_START, VAL_END, OOS_START, OOS_2024_END,
)

BARS_PER_DAY = fl.BARS_PER_DAY
OUT = Path(__file__).resolve().parent


def panel(lb, thr, tf, wall_ms):
    cc, hc = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        held = adx_single_held(np.asarray(c, float), lb, thr)
        cc[sym] = pd.Series(c, index=dates)
        hc[sym] = pd.Series(held, index=dates)
    if len(cc) < 10:
        return None, None
    cdf = pd.DataFrame(cc).sort_index()
    hdf = pd.DataFrame(hc).reindex(cdf.index).fillna(0).astype(np.int8)
    return cdf, hdf


def eval_cell(lb, thr, tf, regime_mode, n_null=100):
    """regime_mode: 'leaked' (current code) | 'deleak' (regime.shift(1))."""
    bpd = BARS_PER_DAY[tf]
    reg_dev = btc_regime(DEV_WALL_MS); reg_oos = btc_regime(OOS_WALL_MS)
    if regime_mode == "deleak":
        reg_dev = reg_dev.shift(1); reg_oos = reg_oos.shift(1)
    cdf, hdf = panel(lb, thr, tf, DEV_WALL_MS)
    if cdf is None:
        return None
    pv = replay(cdf, hdf, reg_dev, VAL_START, VAL_END); mv = profile(pv, bpd)
    vn = float(np.mean(random_null(pv, True, n_null, pv["deploy"]) < mv["compound"])) * 100 if pv["deploy"] > 1e-3 else 0
    cdfo, hdfo = panel(lb, thr, tf, OOS_WALL_MS)
    po = replay(cdfo, hdfo, reg_oos, OOS_START, OOS_2024_END); mo = profile(po, bpd)
    on = float(np.mean(random_null(po, True, n_null, po["deploy"]) < mo["compound"])) * 100 if po["deploy"] > 1e-3 else 0
    return {"val": mv, "val_null": vn, "oos": mo, "oos_null": on}


def main():
    t0 = time.time()
    lbs = [22, 24, 26, 28]
    thrs = [0.42, 0.45, 0.48]
    out = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), "cells": {}}
    for mode in ["leaked", "deleak"]:
        print("=" * 104)
        print(f"ADX|1d SWEEP  regime={mode}  (VAL {VAL_START}..{VAL_END} | OOS {OOS_START}..{OOS_2024_END}, 2025 SEALED)")
        print("=" * 104)
        print(f"  {'lb':>3} {'thr':>5} | {'VALedge':>8} {'VALnull':>7} {'VALdep':>6} | "
              f"{'OOSedge':>8} {'OOSnull':>7} {'OOSdep':>6} | {'shrink%':>7} {'sel':>4} {'cons':>4} {'nullOK':>6}")
        print("  " + "-" * 96)
        for lb in lbs:
            for thr in thrs:
                r = eval_cell(lb, thr, "1d", mode)
                if r is None:
                    continue
                mv, mo, vn, on = r["val"], r["oos"], r["val_null"], r["oos_null"]
                shrink = (1 - mo["edge_pp"] / mv["edge_pp"]) * 100 if abs(mv["edge_pp"]) > 1e-9 else None
                sel = mv["deploy"] < 0.70 and mo["deploy"] < 0.70
                cons = mv["edge_pp"] > 0 and mo["edge_pp"] > 0
                nullok = on >= 95
                out["cells"][f"{mode}_{lb}_{thr}"] = {
                    "mode": mode, "lb": lb, "thr": thr,
                    "val_edge": mv["edge_pp"], "val_null": round(vn, 0), "val_deploy": mv["deploy"],
                    "val_sharpe": mv["sharpe"], "val_pct7d": mv["pct_slices_prof"],
                    "oos_edge": mo["edge_pp"], "oos_null": round(on, 0), "oos_deploy": mo["deploy"],
                    "oos_sharpe": mo["sharpe"], "oos_pct7d": mo["pct_slices_prof"],
                    "oos_lin_wk": mo["lin_wk"], "oos_compound": mo["compound"], "oos_maxdd": mo["maxdd"],
                    "shrink_pct": round(shrink, 1) if shrink is not None else None,
                    "selective": sel, "consistent": cons, "null_ok_95": nullok}
                shp = f"{shrink:>6.0f}%" if shrink is not None else "   NA "
                print(f"  {lb:>3} {thr:>5.2f} | {mv['edge_pp']:>+7.3f} {vn:>6.0f} {mv['deploy']:>5.1%} | "
                      f"{mo['edge_pp']:>+7.3f} {on:>6.0f} {mo['deploy']:>5.1%} | {shp} "
                      f"{'Y' if sel else 'n':>4} {'Y' if cons else 'n':>4} {'Y' if nullok else 'n':>6}")
        print()
    out["elapsed_s"] = round(time.time() - t0, 1)
    p = OUT / "audit_adx.json"
    json.dump(out, open(p, "w"), indent=1, default=str)
    print(f"[json] {p}   [elapsed] {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
