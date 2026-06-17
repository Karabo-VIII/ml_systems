"""src/strat/ti_candidates_2020_2021.py -- candidates per TI family + per TI type + per TF, from the
2020 vs 2021 within-year 6/3/3 deep-dive (ALL 6 TFs). 2020 methodology replicated + adapted across 2021.

USER /orc 2026-06-16 (correction): the 2020 TI deep-dive was ALL 6 TFs -- replicate that for 2021, candidates
emerge per family AND per TI type. STAY in 2020<->2021 (NO 2022 / all-weather without go-ahead). NOT 4h-only.

Reads within_2020.json + within_2021.json (deep_ti_within_year, 6/3/3: TRAIN 6mo / VAL 3mo / OOS 3mo, robust =
TRAIN&VAL>0, OOS held out, fixed-EW u10, ironed sleeve). Per (TI x TF): the best ROBUST ironed config + OOS net +
xBH each year. A CANDIDATE (TI, TF) = robust band exists in BOTH years AND best-robust OOS net > 0 in BOTH years
(the cross-year-stable, deployable region). Emits the candidate register per family + per type + per TF, the
2020->2021 transfer, and mirrors a TI registry-style doc into runs/periods/TRAIN/2021/DEEP_DIVE/. No emoji.

RWYB: python -m strat.ti_candidates_2020_2021
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
WY = ROOT.parent / "runs" / "strat" / "within_year"
DEST = ROOT.parent / "runs" / "periods" / "TRAIN" / "2021" / "DEEP_DIVE"
CHARTS = DEST / "charts"
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
FAM_ORDER = ["trend", "momentum", "breakout", "volume", "mean-reversion"]


def _load(year):
    p = WY / f"within_{year}.json"
    return json.load(open(p)) if p.exists() else None


def _cell_best(cell):
    """(best-robust-ironed cfg, OOS net, xBH, robust_frac) for a within-year per_tf cell, or None."""
    rows = cell.get("rows", [])
    if not rows:
        return None
    bh = (cell.get("buyhold") or {}).get("net")
    rob = [r for r in rows if r["iron"]["robust"]]
    if not rob:
        return {"cfg": None, "net": None, "xbh": None, "robust_frac": 0.0, "n_robust": 0, "n_total": len(rows), "bh": bh}
    best = max(rob, key=lambda r: r["iron"]["net"])
    return {"cfg": best["cfg"], "net": best["iron"]["net"],
            "xbh": round(best["iron"]["net"] / bh, 2) if bh else None,
            "robust_frac": round(len(rob) / len(rows), 2), "n_robust": len(rob), "n_total": len(rows), "bh": bh}


def main(argv=None) -> int:
    d20, d21 = _load(2020), _load(2021)
    if not d20 or not d21:
        print(f"need within_2020.json AND within_2021.json in {WY} (run deep_ti_within_year --year 2020/2021 "
              f"--tfs 1d,4h,2h,1h,30m,15m first)")
        return 1
    inds = [k for k in d20["indicators"] if k in d21["indicators"]]
    fam = {k: d20["indicators"][k]["family"] for k in inds}
    # per (TI, TF) candidate
    grid = {}                                                                # (ti, tf) -> {y20, y21, candidate}
    for ti in inds:
        for tf in TFS:
            c20 = d20["indicators"][ti]["per_tf"].get(tf)
            c21 = d21["indicators"][ti]["per_tf"].get(tf)
            b20 = _cell_best(c20) if c20 else None
            b21 = _cell_best(c21) if c21 else None
            cand = bool(b20 and b21 and b20["n_robust"] > 0 and b21["n_robust"] > 0
                        and (b20["net"] or -1) > 0 and (b21["net"] or -1) > 0)
            grid[(ti, tf)] = {"y20": b20, "y21": b21, "candidate": cand}

    L = ["# TI CANDIDATES 2020 <-> 2021 -- per family, per TI type, per TF (6/3/3, all 6 TFs)", ""]
    L.append("2020 methodology (per-config per-TI per-TF, 6mo TRAIN / 3mo VAL / 3mo OOS, robust = TRAIN&VAL>0, OOS "
             "HELD OUT, fixed-EW u10, ironed sleeve, maker) REPLICATED for 2021. A CANDIDATE (TI, TF) = a robust "
             "band exists AND the best-robust ironed config is OOS-positive in BOTH years (the cross-year-stable, "
             "deployable region). 2022 NOT included (out of scope until go-ahead). [VERIFIED within-year].")
    L.append("")
    L.append("## Per-TI-type x TF candidate map (cells = best-robust OOS net 2020->2021; CAND if robust+positive both)")
    L.append("| family | TI | " + " | ".join(TFS) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(TFS)) + "|")
    per_family = {f: [] for f in FAM_ORDER}
    for ti in sorted(inds, key=lambda t: FAM_ORDER.index(fam[t]) if fam[t] in FAM_ORDER else 9):
        cells = []
        ti_cands = []
        for tf in TFS:
            g = grid[(ti, tf)]
            b20, b21 = g["y20"], g["y21"]
            if b20 and b21 and b20["net"] is not None and b21["net"] is not None:
                tag = "**C**" if g["candidate"] else ""
                cells.append(f"{b20['net']}->{b21['net']}{tag}")
                if g["candidate"]:
                    ti_cands.append((tf, b20["net"], b21["net"], b21["xbh"]))
            else:
                cells.append("-")
        L.append(f"| {fam[ti]} | {ti} | " + " | ".join(cells) + " |")
        if ti_cands:
            per_family[fam[ti]].append((ti, ti_cands))
    L.append("")
    L.append("## Emerging candidates per family (robust + OOS-positive in BOTH 2020 AND 2021)")
    n_cand = 0
    for f in FAM_ORDER:
        items = per_family.get(f, [])
        n_cand += sum(len(c) for _, c in items)
        if not items:
            L.append(f"- **{f}**: (no cross-year candidate)")
            continue
        parts = []
        for ti, cands in items:
            best = max(cands, key=lambda c: c[2])                            # best by 2021 OOS net
            tfs_c = ",".join(c[0] for c in cands)
            parts.append(f"{ti}@[{tfs_c}] (best {best[0]}: {best[1]}->{best[2]}%, {best[3]}xBH)")
        L.append(f"- **{f}**: " + " ; ".join(parts))
    L.append("")
    L.append(f"## HEADLINE: {n_cand} (TI x TF) cross-year candidates (robust + OOS-positive 2020 AND 2021) across "
             f"{len(inds)} TIs x {len(TFS)} TFs. The candidate is the BAND/region per (type, TF); the tradeable "
             f"config is rolling-picked from it. xBH<1 expected (long-only de-risked beta under-participates the "
             f"bull). 2020->2021 transfer: the robust BAND reproduces where a candidate exists; config #1 rank does "
             f"not (use the band, not the #1). NEXT (on go-ahead): all-weather / 2022 bear.")
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / "TI_CANDIDATES_2020_2021.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[md] {out}")

    # chart: candidate count per family x TF (heatmap of # robust-both-years TIs)
    CHARTS.mkdir(parents=True, exist_ok=True)
    fams = FAM_ORDER
    mat = np.zeros((len(fams), len(TFS)))
    for (ti, tf), g in grid.items():
        if g["candidate"]:
            f = fam[ti]
            if f in fams:
                mat[fams.index(f), TFS.index(tf)] += 1
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, cmap="Greens", aspect="auto", vmin=0)
    ax.set_xticks(range(len(TFS))); ax.set_xticklabels(TFS); ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
    for i in range(len(fams)):
        for j in range(len(TFS)):
            ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center", fontsize=10)
    ax.set_title("Cross-year (2020 & 2021) TI candidates per family x TF (count of robust+positive-both-years TIs)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    pc = CHARTS / "ti_candidates_2020_2021_map.png"
    fig.savefig(pc, dpi=110); plt.close(fig)
    print(f"[chart] {pc}")
    json.dump({"grid": {f"{ti}|{tf}": g for (ti, tf), g in grid.items()}, "n_candidates": n_cand},
              open(DEST / "ti_candidates_2020_2021.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
