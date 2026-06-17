"""src/strat/deep2020_conc_law.py -- BLOCK G: does CONCENTRATION predict TIMING SKILL? (the structural law)

Block F found timing skill (low permutation p) only on XRP/DOGE -- the serially-concentrated pump assets.
This tests the LAW: across (instrument, TF), is an asset's timing-alpha predicted by its concentration /
trend-persistence? Joins timingalpha_*.json (timing_alpha, p) with stats_*.json (top10% concentration,
hurst, max_up_run, ac1). If concentration -> timing skill, that is a predictive structural rule: route
trend-following to CONCENTRATED/PUMPY assets, buy-hold the diffuse ones. Scatter + correlations.
RWYB: python -m strat.deep2020_conc_law. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"


def _load(prefix):
    d = {}
    for jf in BASE.glob(f"{prefix}_*.json"):
        for k, v in json.load(open(jf)).items():
            d[k] = v
    return d


def main() -> int:
    ta = _load("timingalpha"); st = _load("stats")
    keys = [k for k in ta if k in st]
    if not keys:
        print("no overlapping (cad|sym) keys between timingalpha and stats -- run those blocks on the same TFs")
        return 1
    rows = []
    for k in keys:
        cad, sym = k.split("|")
        rows.append({"cad": cad, "sym": sym, "alpha_bp": ta[k]["timing_alpha_bp"], "p": ta[k]["p_value"],
                     "conc10": st[k].get("conc_top10"), "hurst": st[k].get("hurst"),
                     "uprun": st[k].get("max_up_run"), "ac1": st[k].get("ac1"),
                     "bigcap": st[k].get("bigbar_capture")})
    rows = [r for r in rows if r["conc10"] is not None and r["alpha_bp"] is not None]

    def corr(x, y):
        x = np.array(x, float); y = np.array(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 3 else float("nan")

    alpha = [r["alpha_bp"] for r in rows]; pval = [r["p"] for r in rows]
    print(f"BLOCK G concentration->timing law -- n={len(rows)} (instrument,TF) cells\n")
    print("## correlation of TIMING-ALPHA (bp) with structural metrics:")
    for met in ["conc10", "hurst", "uprun", "ac1", "bigcap"]:
        print(f"   alpha ~ {met:7}: r = {corr([r[met] for r in rows], alpha):+.2f}    "
              f"p_value ~ {met:7}: r = {corr([r[met] for r in rows], pval):+.2f}")
    # the headline: do the SIGNIFICANT-timing cells have higher concentration?
    sig = [r for r in rows if r["p"] < 0.10]; nons = [r for r in rows if r["p"] >= 0.10]
    print(f"\n## cells with timing skill (p<0.10, n={len(sig)}) vs none (n={len(nons)}):")
    for met in ["conc10", "uprun", "hurst", "bigcap"]:
        sv = np.nanmean([r[met] for r in sig]) if sig else float("nan")
        nv = np.nanmean([r[met] for r in nons]) if nons else float("nan")
        print(f"   mean {met:7}: skill={sv:.3f}  no-skill={nv:.3f}  ({'HIGHER w/ skill' if sv > nv else 'lower w/ skill'})")
    print("\n## the timing-skill cells (p<0.10):")
    for r in sorted(sig, key=lambda r: r["p"]):
        print(f"   {r['cad']:4} {r['sym'].replace('USDT',''):6} alpha {r['alpha_bp']:>7.1f}bp p {r['p']:.3f}  "
              f"conc10 {r['conc10']:.2f} uprun {r['uprun']} hurst {r['hurst']}")

    # scatter
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    c10 = [r["conc10"] for r in rows]
    sc = ax[0].scatter(c10, alpha, c=[("red" if r["p"] < 0.10 else "grey") for r in rows], s=40)
    for r in rows:
        if r["p"] < 0.10:
            ax[0].annotate(r["sym"].replace("USDT", "") + r["cad"], (r["conc10"], r["alpha_bp"]), fontsize=7)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xlabel("concentration (top-10% up-bar share)")
    ax[0].set_ylabel("timing-alpha (bp)"); ax[0].set_title(f"timing-alpha vs concentration (r={corr(c10, alpha):+.2f}; red=p<0.10)")
    ur = [r["uprun"] for r in rows]
    ax[1].scatter(ur, alpha, c=[("red" if r["p"] < 0.10 else "grey") for r in rows], s=40)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("longest up-run (bars)")
    ax[1].set_ylabel("timing-alpha (bp)"); ax[1].set_title(f"timing-alpha vs max up-run (r={corr(ur, alpha):+.2f})")
    fig.tight_layout()
    out = BASE / "charts" / "concentration_law.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
