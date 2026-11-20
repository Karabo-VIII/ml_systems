"""src/strat/ma_analysis_plots.py -- charts + best-config-per-timeframe for the MA building-block analysis.

WHY (user /orc 2026-06-11): *"what is the best config across all assets for each time frame? Also,
do actually plot all of these charts, I would like to have a look at them."* This is the VISUAL read
of the building block we produced (the decoupled MA configs x exit x timeframe sweep, oldest month) --
not a verdict, the starting point made legible.

Reads the latest ma_per_instrument JSON (1215 single-asset cells) and emits:
  - BEST CONFIG PER TIMEFRAME: for each cadence, the (config, exit) with the highest CROSS-ASSET mean
    compound (+ how many assets it was positive on). Printed table + a chart.
  - a 6-panel figure: (A) cadence decay, (B) holding-time discriminator scatter, (C) per-instrument
    mean, (D) best config per timeframe, (E) exit x timeframe, (F) winner/loser hold histogram.

RWYB:  python -m strat.ma_analysis_plots
No emoji (cp1252).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT.parent / "runs" / "strat" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
CADENCE_ORDER = ["4h", "1h", "30m", "15m"]
CAD_COLOR = {"4h": "#1b9e77", "1h": "#7570b3", "30m": "#d95f02", "15m": "#e7298a"}


def best_config_per_tf(cells, min_assets=4):
    """For each cadence, rank (config, exit) by cross-asset MEAN compound (>= min_assets present)."""
    agg = defaultdict(list)
    for c in cells:
        agg[(c["cadence"], c["config"], c["exit"])].append(c)
    by_cad = defaultdict(list)
    for (cad, cfg, ex), rs in agg.items():
        comps = [r["compound_pct"] for r in rs]
        if len(comps) < min_assets:
            continue
        by_cad[cad].append({
            "config": cfg, "exit": ex, "n_assets": len(rs),
            "mean": round(float(np.mean(comps)), 2), "median": round(float(np.median(comps)), 2),
            "n_pos": int(sum(x > 0 for x in comps)),
            "mean_hold_bars": round(float(np.mean([r["avg_hold_bars"] for r in rs])), 1),
            "mean_trades": round(float(np.mean([r["n_trades"] for r in rs])), 1),
        })
    for cad in by_cad:
        by_cad[cad].sort(key=lambda x: x["mean"], reverse=True)
    return by_cad


def make_figure(cells, by_cad, out_png):
    comp = np.array([c["compound_pct"] for c in cells])
    hold = np.array([c["avg_hold_bars"] for c in cells])
    cad = np.array([c["cadence"] for c in cells])

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("MA building block -- decoupled 2MA/3MA x exit x timeframe -- u10, oldest month (2020-01-07..02-07)",
                 fontsize=13, fontweight="bold")

    # A. cadence decay: mean compound (bars) + % positive (line)
    a = ax[0, 0]
    means = [comp[cad == c].mean() for c in CADENCE_ORDER]
    pos = [100 * (comp[cad == c] > 0).mean() for c in CADENCE_ORDER]
    a.bar(CADENCE_ORDER, means, color=[CAD_COLOR[c] for c in CADENCE_ORDER], alpha=0.8)
    a.axhline(0, color="k", lw=0.7); a.set_ylabel("mean compound %")
    a2 = a.twinx(); a2.plot(CADENCE_ORDER, pos, "o-", color="black", lw=2); a2.set_ylabel("% of configs positive")
    a2.set_ylim(40, 100)
    a.set_title("A. Finer timeframe -> lower return, fewer winners")

    # B. holding-time discriminator scatter
    b = ax[0, 1]
    for c in CADENCE_ORDER:
        m = cad == c
        b.scatter(hold[m], comp[m], s=14, alpha=0.45, color=CAD_COLOR[c], label=c)
    b.set_xscale("symlog"); b.axhline(0, color="k", lw=0.7)
    b.set_xlabel("avg holding time (bars, symlog)"); b.set_ylabel("compound %")
    b.legend(title="cadence", fontsize=8); b.set_title("B. Winners hold longer (the discriminator)")

    # C. per-instrument mean
    c_ax = ax[0, 2]
    byA = defaultdict(list)
    for cc in cells:
        byA[cc["asset"]].append(cc["compound_pct"])
    assets = sorted(byA, key=lambda a_: np.mean(byA[a_]), reverse=True)
    vals = [np.mean(byA[a_]) for a_ in assets]
    c_ax.barh([a_.replace("USDT", "") for a_ in assets][::-1], vals[::-1],
              color=["#2c7fb8" if v >= 0 else "#cb181d" for v in vals[::-1]])
    c_ax.axvline(0, color="k", lw=0.7); c_ax.set_xlabel("mean compound %")
    c_ax.set_title("C. Mean return per instrument")

    # D. best config per timeframe
    d = ax[1, 0]
    labels, vals, cols = [], [], []
    for c in CADENCE_ORDER:
        if by_cad.get(c):
            top = by_cad[c][0]
            labels.append(f"{c}\n{top['config']}/{top['exit']}")
            vals.append(top["mean"]); cols.append(CAD_COLOR[c])
    d.bar(range(len(labels)), vals, color=cols, alpha=0.85)
    d.set_xticks(range(len(labels))); d.set_xticklabels(labels, fontsize=8)
    d.axhline(0, color="k", lw=0.7); d.set_ylabel("cross-asset mean %")
    d.set_title("D. Best config per timeframe (cross-asset mean)")

    # E. exit x cadence
    e = ax[1, 1]
    exits = ["signalflip", "trail5", "trail10"]
    width = 0.25
    x = np.arange(len(CADENCE_ORDER))
    for i, ex in enumerate(exits):
        ev = [np.mean([cc["compound_pct"] for cc in cells if cc["cadence"] == c and cc["exit"] == ex] or [0])
              for c in CADENCE_ORDER]
        e.bar(x + (i - 1) * width, ev, width, label=ex)
    e.set_xticks(x); e.set_xticklabels(CADENCE_ORDER); e.axhline(0, color="k", lw=0.7)
    e.set_ylabel("mean compound %"); e.legend(fontsize=8); e.set_title("E. Exit mechanism x timeframe")

    # F. winner vs loser hold histogram
    f = ax[1, 2]
    top_q = comp >= np.percentile(comp, 75); bot_q = comp <= np.percentile(comp, 25)
    bins = np.logspace(0, np.log10(max(hold.max(), 2)), 24)
    f.hist(np.clip(hold[top_q], 1, None), bins=bins, alpha=0.6, color="#2c7fb8", label="winners (top 25%)")
    f.hist(np.clip(hold[bot_q], 1, None), bins=bins, alpha=0.6, color="#cb181d", label="losers (bottom 25%)")
    f.set_xscale("log"); f.set_xlabel("avg holding time (bars, log)"); f.set_ylabel("# cells")
    f.legend(fontsize=8); f.set_title("F. Holding time: winners vs losers")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> int:
    files = sorted(glob.glob(str(ROOT.parent / "runs" / "strat" / "ma_per_instrument_u10_*.json")))
    if not files:
        print("no ma_per_instrument JSON found; run strat.ma_per_instrument first"); return 2
    d = json.load(open(files[-1]))
    cells = d["cells"]
    by_cad = best_config_per_tf(cells)

    print(f"## BEST CONFIG PER TIMEFRAME (across all assets) -- {len(cells)} cells, window {d.get('window')}")
    print(f"   {'tf':4} {'config':16} {'exit':10} {'xasset_mean%':>12} {'median%':>8} {'n_pos/n':>8} {'hold(b)':>8} {'trades':>7}")
    for cad in CADENCE_ORDER:
        for t in by_cad.get(cad, [])[:3]:
            star = " <== BEST" if t is by_cad[cad][0] else ""
            print(f"   {cad:4} {t['config']:16} {t['exit']:10} {t['mean']:>12} {t['median']:>8} "
                  f"{str(t['n_pos'])+'/'+str(t['n_assets']):>8} {t['mean_hold_bars']:>8} {t['mean_trades']:>7}{star}")

    out = PLOTS / "ma_building_block_analysis.png"
    make_figure(cells, by_cad, out)
    print(f"\n[figure] {out}")
    # persist the best-config table
    bc = {"window": d.get("window"), "best_config_per_tf": {c: by_cad.get(c, []) for c in CADENCE_ORDER}}
    p = PLOTS.parent / "ma_best_config_per_tf.json"
    json.dump(bc, open(p, "w", encoding="utf-8"), indent=1)
    print(f"[table]  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
