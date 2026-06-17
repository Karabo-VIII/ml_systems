"""src/strat/deep2020_dashboard.py -- the 2020 deep-dive DASHBOARD (one view of all blocks).

Reads the deep-dive JSONs and renders a 4-panel summary: (A) the STRATEGY LADDER (net% + Sharpe for
MA_FAMILY / BUYHOLD / VOLTGT / XS_MOM, with their honest nature); (B) PARTICIPATION (capt/BH by MA x
cadence); (C) the EXIT DIAL (net vs maxDD across exits); (D) the CONCENTRATION->TIMING law scatter.
RWYB: python -m strat.deep2020_dashboard. No emoji (cp1252).
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
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]
CADS = ["1d", "4h", "2h", "1h", "30m", "15m"]


def _load(name):
    p = BASE / name
    return json.load(open(p)) if p.exists() else {}


def main() -> int:
    opt = _load("optimal_1d_4h_2h_1h.json"); vt = _load("voltarget.json"); xs = _load("xsection.json")
    part = {}
    for f in BASE.glob("participation_*.json"):
        part.update(json.load(open(f)))
    exits = {}
    for f in BASE.glob("exits_*.json"):
        exits.update(json.load(open(f)))
    fig = plt.figure(figsize=(17, 11))

    # (A) strategy ladder @ 1d (net + Sharpe), honest labels
    axA = fig.add_subplot(2, 2, 1)
    lad = [("MA_FAMILY", opt.get("1d|MA_FAMILY", {}).get("net"), None, "de-risked BH\n(no timing skill)"),
           ("BUYHOLD", opt.get("1d|BUYHOLD", {}).get("net"), vt.get("1d|BUYHOLD", {}).get("sharpe"), "the DRIFT\n(optimal participation)"),
           ("VOLTGT_hi", vt.get("1d|VOLTGT_hi", {}).get("net"), vt.get("1d|VOLTGT_hi", {}).get("sharpe"), "vol-targeted BH\n(best risk-adj; ROBUST)"),
           ("XS_MOM", xs.get("1d|XS_MOM", {}).get("net"), xs.get("1d|XS_MOM", {}).get("sharpe"), "bull-beta tilt\n(in-sample; weak persist)")]
    names = [x[0] for x in lad]; nets = [x[1] if x[1] is not None else np.nan for x in lad]
    bars = axA.bar(names, nets, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"])
    for b, x in zip(bars, lad):
        axA.annotate(x[3], (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=7)
    axA.set_ylabel("2020 H2 net % (1d, equal-weight book)")
    axA.set_title("(A) STRATEGY LADDER -- everything is a TILT on the drift; only vol-target robustly improves it")

    # (B) participation capt/BH by MA x cadence
    axB = fig.add_subplot(2, 2, 2)
    M = np.full((len(MA_TYPES), len(CADS)), np.nan)
    for i, mt in enumerate(MA_TYPES):
        for j, cad in enumerate(CADS):
            vals = [v.get("capture_vs_bh") for k, v in part.items() if k.startswith(f"{cad}|{mt}|") and v.get("capture_vs_bh") is not None]
            if vals:
                M[i, j] = float(np.mean(vals))
    im = axB.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1.3)
    axB.set_xticks(range(len(CADS))); axB.set_xticklabels(CADS); axB.set_yticks(range(len(MA_TYPES))); axB.set_yticklabels(MA_TYPES)
    for i in range(len(MA_TYPES)):
        for j in range(len(CADS)):
            if np.isfinite(M[i, j]):
                axB.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    axB.set_title("(B) PARTICIPATION capt/BH (MA x cadence) -- <1 = under-rides the bull (green=beats BH)")

    # (C) exit dial: net vs maxDD across exits @ 1d
    axC = fig.add_subplot(2, 2, 3)
    exrows = {}
    for k, v in exits.items():
        cad, ex, sym = k.split("|")
        if cad != "1d":
            continue
        exrows.setdefault(ex, []).append(v)
    for ex, rows in exrows.items():
        n = np.nanmean([r["net_pct"] for r in rows if r.get("net_pct") is not None])
        d = np.nanmean([r["maxdd_pct"] for r in rows if r.get("maxdd_pct") is not None])
        axC.scatter(d, n, s=50); axC.annotate(ex, (d, n), fontsize=7)
    axC.set_xlabel("maxDD % (less negative = safer)"); axC.set_ylabel("net %")
    axC.set_title("(C) EXIT DIAL @1d -- looser exits (flip/minhold) = more return+DD; the return<->DD frontier")

    # (D) concentration law (recompute from timingalpha+stats)
    axD = fig.add_subplot(2, 2, 4)
    ta = {}; st = {}
    for f in BASE.glob("timingalpha_*.json"):
        ta.update(json.load(open(f)))
    for f in BASE.glob("stats_*.json"):
        st.update(json.load(open(f)))
    xs_c = []; ys = []; cols = []
    for k in ta:
        if k in st and st[k].get("conc_top10") is not None:
            xs_c.append(st[k]["conc_top10"]); ys.append(ta[k]["timing_alpha_bp"]); cols.append("red" if ta[k]["p_value"] < 0.10 else "grey")
    axD.scatter(xs_c, ys, c=cols, s=40); axD.axhline(0, color="k", lw=0.6)
    axD.set_xlabel("concentration (top-10% up-bar share)"); axD.set_ylabel("timing-alpha (bp)")
    r = float(np.corrcoef(xs_c, ys)[0, 1]) if len(xs_c) > 3 else float("nan")
    axD.set_title(f"(D) LAW: timing skill ~ concentration (r={r:+.2f}; red=p<0.10) -- TS edge only on pumps")

    fig.suptitle("2020 DEEP DIVE -- the edge is the DRIFT-BETA; timing/selection are tilts, only vol-target robustly improves; "
                 "TS-timing edge only on concentrated pumps", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = BASE / "charts" / "deep2020_dashboard.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
