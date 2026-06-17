"""src/strat/ma_move_grid.py -- per-instrument, per-timeframe price charts with the BEST 2MA + BEST 3MA drawn on.

WHY (user /orc 2026-06-11): *"I want plots per instrument, per time frame, and the best 2MA and 3MA
configs plotted on those graphs."* + *"I only asked for the starting Jan month"* (so: STRICTLY the
oldest month, explicit date axis).

For each instrument, one figure with a panel per timeframe (4h/1h/30m/15m). Each panel = the Jan-month
log price + the BEST 2MA config (blue) and the BEST 3MA config (orange) for that (instrument, timeframe),
each with entry (^) / exit (v) markers + a light shaded held span. "Best" = highest compound at the
signal-flip exit (pure MA cross) so the chart is about the MA CONFIG itself. xlim is pinned to the
window so nothing spills beyond Jan.

Reuses holding_state + per_asset_trades + ma_per_instrument._panel. RWYB: python -m strat.ma_move_grid
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

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, TAKER_RT
from strat.portfolio_replay_per_asset import per_asset_trades
from strat.replay_distinct_grid import distinct_specs
from strat.ma_per_instrument import _panel

PLOTS = ROOT.parent / "runs" / "strat" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
START, END = "2020-01-07", "2020-02-07"
TFS = ["4h", "1h", "30m", "15m"]
FAM_STYLE = {"2MA": ("#1f77b4", "best 2MA"), "3MA": ("#ff7f0e", "best 3MA")}


def _overlay(ax, name, o, c, ms, s_ms, e_ms, color):
    """Draw one config's entries/exits + held span on the price axis (window only)."""
    held = holding_state(name, o, c, c, c).astype(np.int8)   # 2MA/3MA use close only
    trades = [t for t in per_asset_trades(o, c, held, ms, TAKER_RT) if s_ms <= t["entry_ms"] < e_ms]
    wmask = ms >= s_ms
    dts = pd.to_datetime(ms[wmask], unit="ms").to_numpy()
    hw = held[wmask]
    in_span, x0 = False, None
    for i in range(len(hw)):
        if hw[i] and not in_span:
            in_span, x0 = True, dts[i]
        elif not hw[i] and in_span:
            ax.axvspan(x0, dts[i], color=color, alpha=0.08); in_span = False
    if in_span:
        ax.axvspan(x0, dts[-1], color=color, alpha=0.08)
    for t in trades:
        ei, xi = t["entry_idx"], t["exit_idx"]
        if ms[ei] >= s_ms:
            ax.scatter(pd.to_datetime(ms[ei], unit="ms"), c[ei], marker="^", color=color, s=46, zorder=6, edgecolors="k", linewidths=0.4)
        ax.scatter(pd.to_datetime(ms[xi], unit="ms"), c[xi], marker="v", color=color, s=46, zorder=6, edgecolors="k", linewidths=0.4)
    return len(trades)


def main() -> int:
    cells = json.load(open(sorted(glob.glob(str(ROOT.parent / "runs" / "strat" / "ma_per_instrument_u10_*.json")))[-1]))["cells"]
    sf = [c for c in cells if c["exit"] == "signalflip"]
    best = {}                                              # (asset, tf, family) -> best cell
    for c in sf:
        k = (c["asset"], c["cadence"], c["family"])
        if k not in best or c["compound_pct"] > best[k]["compound_pct"]:
            best[k] = c
    specs = {}
    for fam in ("2MA", "3MA"):
        specs.update(distinct_specs(fam, 0.15, max_n=2000))   # FULL distinct set so any best config resolves
    PR.STRATS.update(specs)
    s_ms = pd.Timestamp(START).value // 10**6
    e_ms = pd.Timestamp(END).value // 10**6
    s_dt, e_dt = pd.Timestamp(START), pd.Timestamp(END)
    assets = sorted({c["asset"] for c in cells})
    files = []
    for asset in assets:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        fig.suptitle(f"{asset.replace('USDT','')} -- best 2MA (blue) & best 3MA (orange) per timeframe -- "
                     f"oldest month {START}..{END}", fontsize=12, fontweight="bold")
        for ax, tf in zip(axes.flatten(), TFS):
            try:
                o, h, l, c, ms = _panel(asset, tf)
            except Exception:
                ax.set_title(f"{tf}: no data"); continue
            keep = ms < e_ms
            o, c, ms = o[keep], c[keep], ms[keep]
            wmask = ms >= s_ms
            if wmask.sum() < 5:
                ax.set_title(f"{tf}: <5 bars in window"); continue
            ax.plot(pd.to_datetime(ms[wmask], unit="ms"), c[wmask], color="#555", lw=0.8, zorder=2)
            ax.set_yscale("log")
            lbls = []
            for fam in ("2MA", "3MA"):
                cell = best.get((asset, tf, fam))
                color, _ = FAM_STYLE[fam]
                if cell is None:
                    continue
                ntr = _overlay(ax, cell["config"], o, c, ms, s_ms, e_ms, color)
                lbls.append(f"{fam} {cell['config']} = {cell['compound_pct']:+.1f}% ({ntr}tr)")
            ax.set_xlim(s_dt, e_dt)                          # PIN to the oldest month -- no spill
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
            ax.set_title(f"{tf}   |   " + "   ".join(lbls), fontsize=8)
            ax.tick_params(labelsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = PLOTS / f"ma_moves_{asset.replace('USDT','')}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        files.append(str(out))
        print(f"[figure] {out}")
    print("FILES:", " ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
