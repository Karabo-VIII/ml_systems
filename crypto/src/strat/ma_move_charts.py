"""src/strat/ma_move_charts.py -- per-asset PRICE charts with the trades drawn on (the visual building block).

WHY (user /orc 2026-06-11): *"do actually plot all of these charts, I would like to have a look."* This
shows, per instrument, the BEST (config, exit, timeframe) from the MA sweep DRAWN ON THE PRICE: log
price over the oldest month + entry (^) / exit (v) markers + shaded held spans. So the aggregates in
ma_analysis_plots become concrete moves you can eyeball.

Reuses the engine: holding_state (signal), apply_trail_stop (mechanical exit), per_asset_trades (round
trips), and ma_per_instrument._panel (load + floor + dedup). Per asset, the best cell (max compound) is
selected from the latest ma_per_instrument JSON.

RWYB:  python -m strat.ma_move_charts
No emoji (cp1252).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, apply_trail_stop, TAKER_RT
from strat.portfolio_replay_per_asset import per_asset_trades
from strat.replay_distinct_grid import distinct_specs
from strat.ma_per_instrument import _panel

PLOTS = ROOT.parent / "runs" / "strat" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
START, END = "2020-01-07", "2020-02-07"


def main() -> int:
    files = sorted(glob.glob(str(ROOT.parent / "runs" / "strat" / "ma_per_instrument_u10_*.json")))
    if not files:
        print("run strat.ma_per_instrument first"); return 2
    cells = json.load(open(files[-1]))["cells"]
    # best cell per asset (max compound)
    best = {}
    for c in cells:
        b = best.get(c["asset"])
        if b is None or c["compound_pct"] > b["compound_pct"]:
            best[c["asset"]] = c
    # need the distinct specs in STRATS to resolve configs by name
    specs = {}
    for fam in ("2MA", "3MA"):
        specs.update(distinct_specs(fam, 0.15, max_n=8))
    PR.STRATS.update(specs)

    s_ms = pd.Timestamp(START).value // 10**6
    e_ms = pd.Timestamp(END).value // 10**6
    assets = sorted(best, key=lambda a: best[a]["compound_pct"], reverse=True)
    n = len(assets)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(17, 3.0 * ((n + 1) // 2)))
    fig.suptitle(f"MA building block -- each instrument's BEST config drawn on price (oldest month {START}..{END})",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()
    trail_of = {"signalflip": 0.0, "trail5": 0.05, "trail10": 0.10}

    for ax, asset in zip(axes, assets):
        cell = best[asset]
        name, ex, cad = cell["config"], cell["exit"], cell["cadence"]
        try:
            o, h, l, c, ms = _panel(asset, cad)
        except Exception as e:
            ax.set_title(f"{asset}: load failed"); continue
        keep = ms < e_ms
        o, c, ms = o[keep], c[keep], ms[keep]
        held = holding_state(name, o, h[keep] if len(h) == len(keep) else c, l[keep] if len(l) == len(keep) else c, c).astype(np.int8)
        tr = trail_of[ex]
        if tr > 0:
            held = apply_trail_stop(held.copy(), c, tr)[0].astype(np.int8)
        trades = [t for t in per_asset_trades(o, c, held, ms, TAKER_RT) if s_ms <= t["entry_ms"] < e_ms]
        # plot window slice
        wmask = ms >= s_ms
        dates = pd.to_datetime(ms[wmask], unit="ms")
        ax.plot(dates, c[wmask], color="#444", lw=0.9)
        ax.set_yscale("log")
        # shade held spans (within window)
        idx_w = np.where(wmask)[0]
        held_w = held[wmask]
        in_span = False; x0 = None
        dts = dates.to_numpy()
        for i in range(len(held_w)):
            if held_w[i] and not in_span:
                in_span = True; x0 = dts[i]
            elif not held_w[i] and in_span:
                ax.axvspan(x0, dts[i], color="#2c7fb8", alpha=0.12); in_span = False
        if in_span:
            ax.axvspan(x0, dts[-1], color="#2c7fb8", alpha=0.12)
        # entry/exit markers
        for t in trades:
            ei, xi = t["entry_idx"], t["exit_idx"]
            if ei < len(c) and ms[ei] >= s_ms:
                ax.scatter(pd.to_datetime(ms[ei], unit="ms"), c[ei], marker="^", color="green", s=40, zorder=5)
            if xi < len(c):
                ax.scatter(pd.to_datetime(ms[xi], unit="ms"), c[xi], marker="v", color="red", s=40, zorder=5)
        ax.set_title(f"{asset.replace('USDT','')}  {name}/{ex}/{cad}  = {cell['compound_pct']}%  "
                     f"({len(trades)}tr)", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS / "ma_move_charts_best_per_asset.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
