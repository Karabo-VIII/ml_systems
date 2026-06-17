"""src/strat/deep2020_xs_mechanism.py -- BLOCK I: WHY does cross-sectional momentum work in 2020?

Block H/H2 found robust XS momentum. This opens the mechanism: does an asset's RELATIVE RANK persist?
  RANK PERSISTENCE -- Spearman corr of (trailing-week return rank) with (next-week return rank), averaged
                      over rebalances. >0 = winners keep winning (momentum); <0 = reversal.
  WINNER-MINUS-LOSER -- avg next-week return of the trailing top-3 minus the trailing bottom-3 (the spread
                      the XS strategy harvests).
  DISPERSION over time -- is there persistently enough cross-sectional spread to exploit.
RWYB: python -m strat.deep2020_xs_mechanism. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.deep2020_xsection import _panel_df, WIN
from scipy.stats import spearmanr


def main() -> int:
    out = {}
    for cad, lb in [("1d", 7), ("4h", 42)]:
        df = _panel_df(cad)
        df = df[df.index >= pd.Timestamp(WIN[0]) - pd.Timedelta(days=10)]
        # forward & trailing lb-bar returns
        trail = df.pct_change(lb)
        fwd = df.shift(-lb) / df - 1.0
        idx = df.index
        persist = []; wml = []; disp = []
        for i in range(lb, len(idx) - lb, max(1, lb // 2)):     # step ~half a window
            tr = trail.iloc[i].dropna(); fw = fwd.iloc[i].dropna()
            common = tr.index.intersection(fw.index)
            if len(common) < 5:
                continue
            tr = tr[common]; fw = fw[common]
            rho = spearmanr(tr.values, fw.values).correlation
            if np.isfinite(rho):
                persist.append(rho)
            top = tr.nlargest(3).index; bot = tr.nsmallest(3).index
            wml.append(float(fw[top].mean() - fw[bot].mean()))
            disp.append(float(df.pct_change().iloc[i].std()))
        mp = float(np.mean(persist)) if persist else float("nan")
        # significance of rank persistence (t on the per-rebalance rhos)
        tstat = float(np.mean(persist) / (np.std(persist) / np.sqrt(len(persist)) + 1e-12)) if len(persist) > 2 else float("nan")
        out[cad] = {"rank_persistence_spearman": round(mp, 3), "persist_tstat": round(tstat, 2),
                    "winner_minus_loser_fwd_pct": round(float(np.mean(wml)) * 100, 2),
                    "wml_frac_positive": round(float(np.mean(np.array(wml) > 0)), 2),
                    "mean_dispersion_pct": round(float(np.mean(disp)) * 100, 2), "n_rebal": len(wml)}
        print(f"########## {cad} (trailing/forward {lb} bars) ##########")
        print(f"   rank persistence (Spearman trail->fwd): {out[cad]['rank_persistence_spearman']:+.3f}  "
              f"t={out[cad]['persist_tstat']:+.2f}  (>0 = winners persist)")
        print(f"   winner-minus-loser next-window return: {out[cad]['winner_minus_loser_fwd_pct']:+.2f}%  "
              f"(positive in {out[cad]['wml_frac_positive']*100:.0f}% of {out[cad]['n_rebal']} rebalances)")
        print(f"   mean cross-sectional dispersion: {out[cad]['mean_dispersion_pct']:.2f}%/bar\n")

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op / "xs_mechanism.json", "w"), indent=1, default=str)
    print(f"[json] {op / 'xs_mechanism.json'}")
    print("\nMECHANISM: positive rank-persistence + positive winner-minus-loser = momentum is REAL "
          "(relative winners keep winning); the XS edge harvests that persistence x the dispersion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
