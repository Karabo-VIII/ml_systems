"""src/strat/ma_killers.py -- the CONSISTENT strategy-killers in the MA building block.

WHY (user /orc 2026-06-12): "are there consistent patterns that kill strategies? 15m might be whipsawed,
but 30m and 1h should close that gap. A cooldown after a stop-loss would help. But tell me the RESULTS
first." So this is a DIAGNOSIS pass over the full mechanics set (137,766 cells, ma_mechanics.json):
what features consistently predict a LOSING config, is the killer the cadence or the config, does the
30m/1h gap close, and how much of the damage is whipsaw (the thing a cooldown would attack).

Reads ma_mechanics.json (asset,cadence,config,exit,net_pct,gross_pct,cost_drag_pct,n_trades,
trades_per_day,whipsaw_frac,avg_hold_bars,win_rate,family). Parses the MA periods from the config name
-> SLOW period (the trend filter; short slow = frequent crosses = the suspected killer). DESCRIPTIVE.

RWYB: python -m strat.ma_killers
No emoji (cp1252).
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT.parent / "runs" / "strat" / "plots"
TFS = ["4h", "1h", "30m", "15m"]


def _slow(cfg):
    nums = [int(x) for x in re.findall(r"\d+", cfg)]
    return max(nums) if nums else 0


def _fast(cfg):
    nums = [int(x) for x in re.findall(r"\d+", cfg)]
    return min(nums) if nums else 0


def _bucket(slow):
    if slow < 20: return "fast(<20)"
    if slow < 60: return "mid(20-60)"
    if slow < 150: return "slow(60-150)"
    return "vslow(>=150)"


BUCKETS = ["fast(<20)", "mid(20-60)", "slow(60-150)", "vslow(>=150)"]


def main() -> int:
    f = sorted(glob.glob(str(ROOT.parent / "runs" / "strat" / "ma_mechanics.json")))
    if not f:
        print("run strat.ma_mechanics first (taker)"); return 2
    cells = json.load(open(f[-1]))
    for c in cells:
        c["slow"] = _slow(c["config"]); c["fast"] = _fast(c["config"]); c["bucket"] = _bucket(c["slow"])
    net = np.array([c["net_pct"] for c in cells])

    # 1. THE CORE KILLER MAP: net by cadence x slow-MA bucket
    print(f"## CONSISTENT KILLERS -- {len(cells)} cells (full mechanics, taker)\n")
    print("[1] net% by CADENCE x SLOW-MA bucket  (is the killer the cadence or the config?)")
    print(f"   {'cadence':8}" + "".join(f"{b:>14}" for b in BUCKETS))
    for cad in TFS:
        row = f"   {cad:8}"
        for b in BUCKETS:
            v = [c["net_pct"] for c in cells if c["cadence"] == cad and c["bucket"] == b]
            row += f"{(np.mean(v) if v else float('nan')):>13.1f}%"
        print(row)
    print("   %positive by same grid:")
    print(f"   {'cadence':8}" + "".join(f"{b:>14}" for b in BUCKETS))
    for cad in TFS:
        row = f"   {cad:8}"
        for b in BUCKETS:
            v = [c["net_pct"] for c in cells if c["cadence"] == cad and c["bucket"] == b]
            row += f"{(100*np.mean(np.array(v)>0) if v else float('nan')):>12.0f}% "
        print(row)

    # 2. THE 30m/1h GAP question: do slow MAs close it?
    print("\n[2] DOES THE 30m/1h GAP CLOSE?  (slow-MA configs only, vs all)")
    for cad in TFS:
        allc = np.array([c["net_pct"] for c in cells if c["cadence"] == cad])
        slowc = np.array([c["net_pct"] for c in cells if c["cadence"] == cad and c["slow"] >= 60])
        print(f"   {cad:4}  ALL configs {allc.mean():6.1f}% ({100*(allc>0).mean():3.0f}% pos)   "
              f"SLOW-only {slowc.mean():6.1f}% ({100*(slowc>0).mean():3.0f}% pos)")

    # 3. LOSER profile vs WINNER profile
    top = net >= np.percentile(net, 75); bot = net <= np.percentile(net, 25)
    def prof(mask, lab):
        cc = [c for i, c in enumerate(cells) if mask[i]]
        print(f"   {lab}: net {np.mean([c['net_pct'] for c in cc]):6.1f}%  slow {np.median([c['slow'] for c in cc]):4.0f}  "
              f"trades {np.mean([c['n_trades'] for c in cc]):4.0f}  whip {100*np.mean([c['whipsaw_frac'] for c in cc]):3.0f}%  "
              f"hold {np.mean([c['avg_hold_bars'] for c in cc]):4.0f}b  win {100*np.mean([c['win_rate'] for c in cc]):3.0f}%  "
              f"costDrag {np.mean([c['cost_drag_pct'] for c in cc]):4.1f}%")
    print("\n[3] WINNER vs LOSER profile (quartiles):")
    prof(top, "WINNERS"); prof(bot, "LOSERS ")

    # 4. WHIPSAW = the cooldown target. How much pure-cost drag do whipsaw trades cause?
    print("\n[4] WHIPSAW / OVERTRADING = the cooldown target")
    for cad in TFS:
        cc = [c for c in cells if c["cadence"] == cad]
        wfrac = np.mean([c["whipsaw_frac"] for c in cc])
        ntr = np.mean([c["n_trades"] for c in cc])
        # whipsaw trades are ~pure cost: n_whip * round-trip; estimate their drag share
        whip_cost = wfrac * ntr * 0.24                 # taker rt 0.24% per whipsaw round trip
        drag = np.mean([c["cost_drag_pct"] for c in cc])
        print(f"   {cad:4}  whip {100*wfrac:3.0f}%  ~{wfrac*ntr:4.1f} whip-trades/cell  "
              f"est whip-cost ~{whip_cost:4.1f}% of {drag:4.1f}% total drag  ({100*whip_cost/drag if drag else 0:2.0f}% of drag)")

    # 5. consistency: is fast-MA a killer at EVERY cadence?
    print("\n[5] IS FAST-MA A KILLER AT EVERY CADENCE? (net of fast(<20) bucket)")
    for cad in TFS:
        v = np.array([c["net_pct"] for c in cells if c["cadence"] == cad and c["bucket"] == "fast(<20)"])
        print(f"   {cad:4}  fast-MA net {v.mean():6.1f}%  ({100*(v>0).mean():3.0f}% pos, n={len(v)})")

    # 6. per-asset
    print("\n[6] PER-ASSET (mean net, %positive, why XRP lags):")
    for a in sorted({c["asset"] for c in cells}):
        v = np.array([c["net_pct"] for c in cells if c["asset"] == a])
        print(f"   {a.replace('USDT',''):5}  net {v.mean():6.1f}%  {100*(v>0).mean():3.0f}% pos")

    # FIGURE: the killer map (net heatmap, cadence x slow-bucket) + whipsaw share of drag
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    M = np.array([[np.mean([c["net_pct"] for c in cells if c["cadence"] == cad and c["bucket"] == b] or [np.nan])
                   for b in BUCKETS] for cad in TFS])
    im = ax[0].imshow(M, cmap="RdYlGn", vmin=-25, vmax=25, aspect="auto")
    ax[0].set_xticks(range(len(BUCKETS))); ax[0].set_xticklabels(BUCKETS, fontsize=9, rotation=15)
    ax[0].set_yticks(range(len(TFS))); ax[0].set_yticklabels(TFS)
    ax[0].set_xlabel("slow-MA period (trade frequency)"); ax[0].set_ylabel("cadence")
    for i in range(len(TFS)):
        for j in range(len(BUCKETS)):
            ax[0].text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax[0].set_title("THE KILLER MAP: net% by cadence x MA-speed\n(red = dies; the diagonal = optimal slow scales with cadence)")
    fig.colorbar(im, ax=ax[0], label="net %", fraction=0.046)
    # whipsaw share of cost drag by cadence
    wfr = [np.mean([c["whipsaw_frac"] for c in cells if c["cadence"] == cad]) for cad in TFS]
    drag = [np.mean([c["cost_drag_pct"] for c in cells if c["cadence"] == cad]) for cad in TFS]
    ntr = [np.mean([c["n_trades"] for c in cells if c["cadence"] == cad]) for cad in TFS]
    whip_cost = [wfr[i] * ntr[i] * 0.24 for i in range(len(TFS))]
    x = np.arange(len(TFS))
    ax[1].bar(x, drag, color="#bdbdbd", label="total cost drag %")
    ax[1].bar(x, whip_cost, color="#cb181d", label="whipsaw share (cooldown target)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(TFS); ax[1].set_ylabel("cost drag %")
    ax[1].legend(fontsize=9); ax[1].set_title("COST DRAG: whipsaw is ~10-16% of it\n(a cooldown attacks the red; the grey is just too many trades)")
    fig.tight_layout()
    out = PLOTS / "ma_killers.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
