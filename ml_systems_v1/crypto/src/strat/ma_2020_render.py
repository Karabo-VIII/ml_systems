"""src/strat/ma_2020_render.py -- merge per-TF ma_2020_breakdown_*.json into the BY-CLASS decomposition.

User /orc 2026-06-12: "break it down BY MA: SMA (1d,4h,2h,1h,30m,15m); EMA ...; HMA ... -- results decomposed
across the board. And convergence and divergence and coverage on the whole."

Produces (no recompute -- reads the per-TF JSONs incl. the synthesized 2h):
  1. the BY-CLASS table (rows = MA class, cols = TF) of OOS compound -- the decomposition across the board;
  2. a COVERAGE heatmap (class x TF);
  3. per-class TF-PROFILE lines (convergence = lines bunched; divergence = lines spread);
  4. DIVERGENCE per TF (spread across classes = "does the MA class matter at this cadence?") +
     per-class TF-sensitivity (std across TFs) + the natural class GROUPS (plain/low-lag/adaptive).
RWYB: python -m strat.ma_2020_render. No emoji (cp1252).
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
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "MA_2020_BREAKDOWN"
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]          # coarse -> fine
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]
GROUPS = {"plain": ["EMA", "SMA", "WMA"], "low-lag": ["HMA", "DEMA", "TEMA"], "adaptive": ["KAMA", "VIDYA"]}


def main() -> int:
    results, winners, oracle = {}, {}, {}
    for jf in sorted(BASE.glob("ma_2020_breakdown_*.json")):
        if "MERGED" in jf.name:
            continue
        d = json.load(open(jf))
        for k, v in d.get("results", {}).items():
            cad, mt = k.split("|"); results[(cad, mt)] = v
        for c, w in d.get("winners", {}).items():
            if w:
                winners[c] = w
        oracle.update(d.get("oracle", {}))

    def oos(cad, mt):
        v = results.get((cad, mt), {}).get("OOS")
        return float(v) if v is not None else np.nan

    M = np.array([[oos(c, t) for c in CADENCES] for t in MA_TYPES])   # rows=class, cols=TF

    # ---- BY-CLASS table (the decomposition across the board) ----
    print("# 2020 MA breakdown -- BY MA CLASS across all timeframes (within-2020 OOS compound %)\n")
    print(f"   {'CLASS':6}" + "".join(f"{c:>7}" for c in CADENCES) + f"{'MEAN':>7}{'STD':>6}{'best@':>7}")
    for i, t in enumerate(MA_TYPES):
        row = M[i]
        bestc = CADENCES[int(np.nanargmax(row))] if np.isfinite(row).any() else "?"
        print(f"   {t:6}" + "".join(f"{(round(v,1) if np.isfinite(v) else '--'):>7}" for v in row) +
              f"{round(float(np.nanmean(row)),1):>7}{round(float(np.nanstd(row)),1):>6}{bestc:>7}")

    # ---- DIVERGENCE per TF + winner ----
    print(f"\n   {'(winner)':6}" + "".join(f"{(winners.get(c,['?'])[0]):>7}" for c in CADENCES))
    print(f"   {'spread':6}" + "".join(f"{round(float(np.nanmax(M[:,j])-np.nanmin(M[:,j])),1):>7}" for j in range(len(CADENCES)))
          + "   <- DIVERGENCE (max-min across classes; high = MA class MATTERS at this cadence)")

    # ---- CONVERGENCE: group means + within/between ----
    print("\n## CONVERGENCE -- class GROUP mean OOS per TF (plain / low-lag / adaptive)")
    print(f"   {'group':9}" + "".join(f"{c:>7}" for c in CADENCES))
    gmeans = {}
    for g, members in GROUPS.items():
        gm = [float(np.nanmean([oos(c, m) for m in members])) for c in CADENCES]
        gmeans[g] = gm
        print(f"   {g:9}" + "".join(f"{round(v,1):>7}" for v in gm))

    # ---- FIGURE: heatmap + class lines + divergence + group lines ----
    fig = plt.figure(figsize=(17, 11))
    # (A) heatmap coverage
    axA = fig.add_subplot(2, 2, 1)
    im = axA.imshow(M, aspect="auto", cmap="RdYlGn", vmin=np.nanmin(M), vmax=np.nanmax(M))
    axA.set_xticks(range(len(CADENCES))); axA.set_xticklabels(CADENCES)
    axA.set_yticks(range(len(MA_TYPES))); axA.set_yticklabels(MA_TYPES)
    for i in range(len(MA_TYPES)):
        for j in range(len(CADENCES)):
            if np.isfinite(M[i, j]):
                axA.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=8,
                         color="black", fontweight="bold")
    axA.set_title("(A) COVERAGE: OOS compound % -- MA class x timeframe (2020 OOS)")
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04, label="OOS %")
    # (B) per-class TF profile lines (convergence/divergence)
    axB = fig.add_subplot(2, 2, 2)
    xs = range(len(CADENCES))
    for i, t in enumerate(MA_TYPES):
        axB.plot(xs, M[i], marker="o", lw=1.6, label=t)
    axB.set_xticks(list(xs)); axB.set_xticklabels(CADENCES); axB.axhline(0, color="k", lw=0.6)
    axB.set_ylabel("OOS compound %"); axB.legend(fontsize=7, ncol=2)
    axB.set_title("(B) per-class TF profile (lines bunched=CONVERGE, spread=DIVERGE)")
    # (C) divergence per TF
    axC = fig.add_subplot(2, 2, 3)
    spread = [float(np.nanmax(M[:, j]) - np.nanmin(M[:, j])) for j in range(len(CADENCES))]
    axC.bar(list(xs), spread, color="#9467bd")
    for j, s in enumerate(spread):
        axC.annotate(winners.get(CADENCES[j], ["?"])[0], (j, s), ha="center", va="bottom", fontsize=8)
    axC.set_xticks(list(xs)); axC.set_xticklabels(CADENCES)
    axC.set_ylabel("max-min OOS across classes"); axC.set_title("(C) DIVERGENCE per TF (does the MA class matter?)")
    # (D) group-mean lines (convergence of the 3 families)
    axD = fig.add_subplot(2, 2, 4)
    for g, gm in gmeans.items():
        axD.plot(xs, gm, marker="s", lw=2.0, label=g)
    axD.set_xticks(list(xs)); axD.set_xticklabels(CADENCES); axD.axhline(0, color="k", lw=0.6)
    axD.set_ylabel("group-mean OOS %"); axD.legend(fontsize=9)
    axD.set_title("(D) CONVERGENCE of class GROUPS (plain / low-lag / adaptive)")
    fig.suptitle("2020 MA-class decomposition across all timeframes -- coverage / convergence / divergence "
                 "(within-2020 OOS, FULL stack)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = BASE / "charts" / "ma_2020_byclass.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    json.dump({"by_class_oos": {t: {c: (None if not np.isfinite(M[i, j]) else round(float(M[i, j]), 1))
                                    for j, c in enumerate(CADENCES)} for i, t in enumerate(MA_TYPES)},
               "divergence_per_tf": {c: round(spread[j], 1) for j, c in enumerate(CADENCES)},
               "group_means": gmeans, "winners": {c: winners.get(c) for c in CADENCES}},
              open(BASE / "ma_2020_byclass.json", "w"), indent=1, default=str)
    print(f"[json] {BASE / 'ma_2020_byclass.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
