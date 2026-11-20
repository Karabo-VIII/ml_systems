"""src/strat/ma_equity_grid.py -- per-instrument x per-timeframe: price + equity + exit-mechanism overlay.

WHY (user /orc 2026-06-12): "(a) add the equity curve beneath each price panel" + "(b) show the best
config at each exit mechanism (signal-flip vs trail) on the same price." This builds both on the
oldest-month MA block: per instrument, per timeframe, a PRICE panel (best 2MA + best 3MA entries, plus
the best-2MA EXITS under signal-flip vs trail-5 vs trail-10 = (b)) and an EQUITY panel beneath ((a),
$1 net), comparing best-2MA across the 3 exits + best-3MA.

Reuses ma_mechanics.extract (cached panels -> fast). Strictly the oldest month, x-axis pinned.
RWYB: python -m strat.ma_equity_grid
No emoji (cp1252).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.ma_mechanics import extract, START, END, TFS

PLOTS = ROOT.parent / "runs" / "strat" / "plots"
S_MS = pd.Timestamp(START).value // 10**6


def _mark(ax, m, marker, color, which):
    """plot entry or exit markers for a config's windowed trades."""
    for t in m["_trades"]:
        idx = t["entry_idx"] if which == "entry" else t["exit_idx"]
        if m["_ms"][idx] >= S_MS:
            ax.scatter(pd.to_datetime(m["_ms"][idx], unit="ms"), m["_c"][idx], marker=marker,
                       color=color, s=42, zorder=6, edgecolors="k", linewidths=0.3)


def main() -> int:
    cells = json.load(open(sorted(glob.glob(str(ROOT.parent / "runs" / "strat" / "ma_per_instrument_u10_*.json")))[-1]))["cells"]
    sf = [c for c in cells if c["exit"] == "signalflip"]
    best = {}
    for c in sf:
        k = (c["asset"], c["cadence"], c["family"])
        if k not in best or c["compound_pct"] > best[k]["compound_pct"]:
            best[k] = c
    s_dt, e_dt = pd.Timestamp(START), pd.Timestamp(END)
    assets = sorted({c["asset"] for c in cells})
    files = []
    for asset in assets:
        fig, axes = plt.subplots(len(TFS), 2, figsize=(16, 3.0 * len(TFS)),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
        fig.suptitle(f"{asset.replace('USDT','')} -- price + exit overlay (left) & equity (right) -- "
                     f"oldest month {START}..{END}", fontsize=12, fontweight="bold")
        for row, tf in enumerate(TFS):
            axp, axe = axes[row]
            b2 = best.get((asset, tf, "2MA")); b3 = best.get((asset, tf, "3MA"))
            if b2 is None:
                axp.set_title(f"{tf}: no 2MA"); continue
            m2 = {ex: extract(asset, tf, b2["config"], tr) for ex, tr in [("sf", 0.0), ("t5", 0.05), ("t10", 0.10)]}
            m3 = extract(asset, tf, b3["config"], 0.0) if b3 else None
            base = m2["sf"]
            wm = base["_ms"] >= S_MS
            dts = pd.to_datetime(base["_ms"][wm], unit="ms")
            # PRICE panel
            axp.plot(dts, base["_c"][wm], color="#666", lw=0.8, zorder=2)
            axp.set_yscale("log")
            _mark(axp, base, "^", "#1f77b4", "entry")                  # 2MA entries
            _mark(axp, base, "v", "#1f77b4", "exit")                   # 2MA signal-flip exits (blue)
            _mark(axp, m2["t5"], "v", "#2ca02c", "exit")               # trail-5 exits (green)
            _mark(axp, m2["t10"], "v", "#d62728", "exit")              # trail-10 exits (red)
            if m3:
                _mark(axp, m3, "^", "#ff7f0e", "entry")                # 3MA entries (orange)
            axp.set_xlim(s_dt, e_dt); axp.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
            axp.set_title(f"{tf}  PRICE  | 2MA {b2['config']} (^entry, v exit: blue=flip grn=trail5 red=trail10)"
                          + (f"  3MA {b3['config']}" if b3 else ""), fontsize=7)
            axp.tick_params(labelsize=7)
            # EQUITY panel ($1 net)
            for ex, col, ls, lab in [("sf", "#1f77b4", "-", f"2MA flip {m2['sf']['net_pct']:+.0f}%"),
                                     ("t5", "#2ca02c", "--", f"2MA trail5 {m2['t5']['net_pct']:+.0f}%"),
                                     ("t10", "#d62728", "--", f"2MA trail10 {m2['t10']['net_pct']:+.0f}%")]:
                e = m2[ex]; axe.plot(e["_dates"][e["_ms"] >= S_MS], e["_eq_net"][e["_ms"] >= S_MS], col, ls=ls, lw=1.3, label=lab)
            if m3:
                axe.plot(m3["_dates"][m3["_ms"] >= S_MS], m3["_eq_net"][m3["_ms"] >= S_MS], "#ff7f0e", lw=1.3,
                         label=f"3MA flip {m3['net_pct']:+.0f}%")
            axe.axhline(1.0, color="k", lw=0.6, ls=":")
            axe.set_xlim(s_dt, e_dt); axe.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
            axe.set_title(f"{tf}  EQUITY ($1 net)", fontsize=8); axe.legend(fontsize=6.5, loc="upper left")
            axe.tick_params(labelsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = PLOTS / f"ma_equity_{asset.replace('USDT','')}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        files.append(str(out)); print(f"[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
