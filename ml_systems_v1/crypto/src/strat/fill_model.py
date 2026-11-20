"""src/strat/fill_model.py -- LD-1: post-hoc cost + fill realism on CanonicalHarness results.

PROVENANCE: ported 2026-06-05 from runs/staging/fill_model_2026_06_04.py. Hardened against the
2026-06-05 apparatus red-audit (docs/APPARATUS_AUDIT_2026_06_05.md):
  - F12 (MEDIUM, FIXED): adverse selection was applied as `gross * (1 - adverse)`, which shrinks the
    MAGNITUDE of both winners AND losers -- i.e. it made losing trades LESS bad (backwards: adverse
    selection should always worsen the realized net). Fixed to a penalty that always subtracts:
    `new_net = gross - adverse * |gross| - cost - fund`. For winners this is identical to the old
    formula (so the "maker collapses" stress result is preserved); for losers it now correctly
    deepens the loss.

Recovers each trade's gross price return (net_pnl + spec.cost_rt + fund_net) -- so the result is
independent of the wrapped harness's cost default (F9): the spec cost is backed out and the mode's
cost re-applied. Then re-applies a fill model: keep each trade with prob p_fill (Monte-Carlo), apply
the adverse-selection penalty, charge the mode's round-trip cost. Reports median / p05 / p95 compound
per window.

CALIBRATION IS PROVISIONAL (flagged): p_fill / adverse_selection come from
config/maker_cost_calibration.yaml (p_fill 0.21-0.40, adverse 0.96-1.00) which the audit flagged as
uncertain. Taker is the solid default; maker modes are pessimistic stress scenarios. Do NOT ship maker
numbers without recalibrating against real live-fill data.
"""
from __future__ import annotations

import numpy as np

MODES = {
    "taker":             dict(cost_rt=0.0024, p_fill=1.00, adverse=0.00),  # realistic spot taker (solid)
    "maker_pessimistic": dict(cost_rt=0.0010, p_fill=0.30, adverse=0.96),  # worst-bucket maker (provisional)
    "ideal_ref":         dict(cost_rt=0.0010, p_fill=1.00, adverse=0.00),  # the old optimistic default (reference)
}


def apply_fill_model(harness, mode: str, n_mc: int = 400, seed: int = 7) -> dict:
    m = MODES[mode]
    res = harness.run()
    spec_cost = float(harness.spec.cost_rt)
    windows = list(harness.WINDOWS)
    rng = np.random.default_rng(seed)
    # recover gross price return: net_pnl = gross - spec_cost - fund_net  ->  gross = net + spec_cost + fund
    trades = [dict(w=t["window"], gross=t["net_pnl"] + spec_cost + t["fund_net"], fund=t["fund_net"])
              for t in res.trades]
    per_window = {w: [] for w in windows}
    deterministic = m["p_fill"] >= 1.0
    iters = 1 if deterministic else n_mc
    for _ in range(iters):
        comp = {w: [] for w in windows}
        for t in trades:
            if deterministic or rng.random() < m["p_fill"]:
                # F12 FIX: adverse selection is a penalty that always worsens the net.
                new_net = t["gross"] - m["adverse"] * abs(t["gross"]) - m["cost_rt"] - t["fund"]
                comp[t["w"]].append(new_net)
        for w in windows:
            arr = comp[w]
            per_window[w].append(float((np.prod(1.0 + np.array(arr)) - 1.0) * 100) if arr else 0.0)
    summary = {}
    for w in windows:
        a = np.array(per_window[w])
        n_w = len([t for t in trades if t["w"] == w])
        summary[w] = dict(median=round(float(np.median(a)), 2),
                          p05=round(float(np.percentile(a, 5)), 2),
                          p95=round(float(np.percentile(a, 95)), 2),
                          n_filled_avg=round(n_w * m["p_fill"], 1))
    return summary


def _rwyb():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd
    from pipeline.chimera_loader import ChimeraLoader
    from wealth_bot.harness import CanonicalHarness

    def to_pandas(loaded):
        df = pd.DataFrame(loaded.to_dict(as_series=False)) if (hasattr(loaded, "to_dict") and not hasattr(loaded, "iloc")) else loaded
        df["date"] = pd.to_datetime(df["date"], unit="ms") if np.issubdtype(df["date"].dtype, np.number) else pd.to_datetime(df["date"])
        return df
    print("[fillmodel RWYB] BTC 1d R12 under each fill mode (median compound %):")
    df = to_pandas(ChimeraLoader().load("BTCUSDT", cadence="1d"))
    h = CanonicalHarness.from_r12_defaults(df, chimera_path="fillmodel_rwyb")
    for mode in MODES:
        s = apply_fill_model(h, mode)
        row = "  ".join(f"{w}={s[w]['median']:+.1f}%(p05 {s[w]['p05']:+.0f})" for w in ["TRAIN", "VAL", "OOS", "UNSEEN"])
        print(f"  {mode:18}: {row}")
    print("[fillmodel RWYB] EXPECT: taker slightly worse than ideal_ref; maker_pessimistic collapses "
          "(adv 0.96 + p_fill 0.30) -> confirms maker execution is dead for us.")


if __name__ == "__main__":
    _rwyb()
