"""src/strat/capture_k3_correct.py -- K3 REGIME-SPECIFICITY test (corrected, 2026-06-20).

The original K3 regime-shuffle confounds regime pool-composition with regime signal.
Correct K3 tests: is the TI's chop edge SPECIFIC TO CHOP, or is it simply that the TI fires
more in bull and some bull bars leak into the chop label?

TWO CORRECT TESTS:
  K3a: WITHIN-REGIME RANDOM NULL: compare TI vs a same-regime random-entry null (already done in main sweep).
        Does the same TI FIRED on bull have a larger raw realized net than on chop?
        (i.e., is the chop edge driven by LEAKAGE of high-bull returns, not genuine chop identification?)
  K3b: FIRE-RATE REGIME DECOMPOSITION: does the TI fire disproportionately in bull vs chop?
        If fire_rate_bull >> fire_rate_chop, the "chop edge" is actually a high-frequency trigger
        leaking bull-regime returns into the chop label.

DEV wall <= 2024-05-15. No emoji.
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
import strat.capture_lab as cl


def regime_specificity(lab, ti, tf="1d", hold=7, min_move=0.03, n_null=300, seed=5):
    """
    K3 CORRECTED: Test whether the chop edge is genuine or a leakage artifact.

    (A) FIRE RATE by regime: if TI fires disproportionately in bull, chop "edge" = bull leakage.
    (B) RAW REALIZED NET by regime: TI-fired chop bars vs TI-fired bull bars -- are they different?
        (If chop bars have similar realized net to bull bars, and the TI fires in both, the chop label
        is just catching some bull bars).
    (C) The WITHIN-REGIME edge is already computed in the main sweep (TI vs same-regime random null).
        What this test adds: FIRE RATE BALANCE and RAW RETURN DECOMPOSITION.
    """
    C = lab["C"]; bpd = fl.BARS_PER_DAY[tf]
    hold_bars = hold * bpd if tf != "1d" else hold
    TIME = cl.time_return_matrix(C, hold_bars)
    MFE = cl.mfe_matrix(C, hold_bars)
    fired = cl.fired_matrix(lab, ti)
    reg = cl.regime_series(lab, tf)
    MFEa = MFE.to_numpy(); TIMEa = TIME.to_numpy(); Ca = C.to_numpy()
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), bool)
    valid[40:n - hold_bars - 1, :] = True
    valid &= np.isfinite(Ca) & np.isfinite(MFEa)
    pool_mask = valid & (MFEa > min_move)
    fired_mask = fired.to_numpy() & pool_mask
    Ra = reg.to_numpy()

    results = {}
    for rg in ("bull", "chop", "bear"):
        rg_mask = (Ra == rg)[:, None]
        valid_rg = pool_mask & rg_mask
        fired_rg = fired_mask & rg_mask
        n_valid = int(valid_rg.sum())
        n_fired = int(fired_rg.sum())
        fire_rate = float(n_fired / n_valid) if n_valid > 0 else np.nan
        # raw realized net for TI-fired bars in this regime
        idx = np.array(np.where(fired_rg)).T
        if len(idx) > 0:
            real = TIMEa[idx[:, 0], idx[:, 1]]
            real = real[np.isfinite(real)]
            raw_net = float(100 * real.mean()) if len(real) > 0 else np.nan
            mfe_vals = MFEa[idx[:, 0], idx[:, 1]]
            mfe_vals = mfe_vals[np.isfinite(mfe_vals)]
            mean_mfe = float(100 * mfe_vals.mean()) if len(mfe_vals) > 0 else np.nan
        else:
            raw_net = mean_mfe = np.nan
        results[rg] = {"n_valid": n_valid, "n_fired": n_fired, "fire_rate": round(fire_rate, 4),
                       "raw_realized_net_pct": round(raw_net, 3) if np.isfinite(raw_net) else None,
                       "mean_mfe_pct": round(mean_mfe, 3) if np.isfinite(mean_mfe) else None}

    # LEAKAGE DIAGNOSIS: if chop raw_net is SIMILAR to bull raw_net, the chop edge is likely a
    # bull-leakage artifact (the same good bars are slightly mis-labeled).
    bull_net = results.get("bull", {}).get("raw_realized_net_pct")
    chop_net = results.get("chop", {}).get("raw_realized_net_pct")
    bear_net = results.get("bear", {}).get("raw_realized_net_pct")
    bull_rate = results.get("bull", {}).get("fire_rate")
    chop_rate = results.get("chop", {}).get("fire_rate")

    # fire rate concentration: if bull_rate >> chop_rate, TI is bull-biased (expected for momentum)
    rate_ratio = float(bull_rate / chop_rate) if (bull_rate and chop_rate and chop_rate > 0) else np.nan

    # if chop raw_net is well above bear and the TI fires in chop at a meaningful rate -> genuine chop catch
    # if chop raw_net is close to bull raw_net -> leakage
    leakage_flag = None
    if all(v is not None for v in [bull_net, chop_net, bear_net]):
        range_ = abs(bull_net - bear_net)
        if range_ > 0.1:
            chop_position = (chop_net - bear_net) / range_
            # if chop is within 20% of bull (position > 0.8), likely leakage
            leakage_flag = bool(chop_position > 0.75)

    return {
        "ti": ti, "tf": tf,
        "regime_stats": results,
        "fire_rate_bull_vs_chop": round(rate_ratio, 2) if np.isfinite(rate_ratio) else None,
        "chop_position_in_bull_bear_range": round((chop_net - bear_net) / max(abs(bull_net - bear_net), 0.01), 2)
            if all(v is not None for v in [bull_net, chop_net, bear_net]) else None,
        "LEAKAGE_FLAG": leakage_flag,
        "verdict": "LEAKAGE" if leakage_flag else ("GENUINE_CHOP" if leakage_flag is False else "UNCLEAR"),
    }


def main():
    import json, datetime
    print("[K3 CORRECTED] Regime-specificity / leakage test")
    labs = {}
    for tf in ["1d"]:
        bpd = fl.BARS_PER_DAY[tf]
        lab = fl.load_wide(n=50, tf=tf, min_bars=400)
        assert lab["C"].index.max() < pd.Timestamp(fl.DEV_END), "WALL"
        labs[tf] = lab

    candidates = ["mom30", "rangepos", "rsi14", "brk14", "mom14", "mom7", "volexp", "accel"]
    print(f"\n  {'ti':10} {'bull_rate':>10} {'chop_rate':>10} {'rate_ratio':>11} "
          f"{'bull_net%':>10} {'chop_net%':>10} {'bear_net%':>10} {'chop_pos':>9} {'VERDICT':>10}")
    all_r = []
    for ti in candidates:
        r = regime_specificity(labs["1d"], ti, tf="1d", hold=7)
        all_r.append(r)
        rs = r["regime_stats"]
        b = rs.get("bull", {}); c = rs.get("chop", {}); br = rs.get("bear", {})
        print(f"  {ti:10} {b.get('fire_rate', 'N/A'):>10}  {c.get('fire_rate', 'N/A'):>9}  "
              f"{r.get('fire_rate_bull_vs_chop', 'N/A'):>10}  "
              f"{b.get('raw_realized_net_pct', 'N/A'):>9}  {c.get('raw_realized_net_pct', 'N/A'):>9}  "
              f"{br.get('raw_realized_net_pct', 'N/A'):>9}  "
              f"{r.get('chop_position_in_bull_bear_range', 'N/A'):>8}  {r.get('verdict', '?'):>10}")

    # save
    runs_dir = Path(__file__).resolve().parents[2] / "runs" / "strat"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = runs_dir / f"capture_k3_correct_{ts}.json"
    with open(out, "w") as fh:
        json.dump(all_r, fh, default=str, indent=2)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
