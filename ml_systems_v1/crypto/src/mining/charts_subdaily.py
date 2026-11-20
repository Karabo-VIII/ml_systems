"""Sub-daily charts for a universe -- simple visual reference, no analysis.

Per asset: one PNG with 4 panels (4h / 1h / 30m / 15m), each showing the most
recent readable window (90d / 21d / 10d / 7d): close line + high-low band + window
%-change in the title. Plus market-wide contact sheets (all assets, one cadence,
small multiples) for at-a-glance scanning.

Output: runs/mining/plots/<universe>_subdaily_<stamp>/
No emoji (cp1252).

Run:
  python -m mining.charts_subdaily --universe u50
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

DAY_MS = 86_400_000
WINDOWS_DAYS = {"4h": 90, "1h": 21, "30m": 10, "15m": 7}
CADENCES = ["4h", "1h", "30m", "15m"]


def load_window(sym: str, cad: str):
    df = ChimeraLoader().load(sym, cadence=cad, features=["high", "low", "close"])
    ts = df["timestamp"].to_numpy()
    cut = ts.max() - WINDOWS_DAYS[cad] * DAY_MS
    m = ts >= cut
    x = [dt.datetime.fromtimestamp(int(t) / 1000, dt.timezone.utc) for t in ts[m]]
    return (x, df["high"].to_numpy()[m].astype(float),
            df["low"].to_numpy()[m].astype(float),
            df["close"].to_numpy()[m].astype(float))


def panel(ax, sym, cad, compact=False):
    try:
        x, h, l, c = load_window(sym, cad)
    except Exception as ex:
        ax.text(0.5, 0.5, f"{cad}: no data ({type(ex).__name__})",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    if len(c) < 5:
        ax.text(0.5, 0.5, f"{cad}: too few bars", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return
    chg = (c[-1] / c[0] - 1) * 100
    color = "tab:green" if chg >= 0 else "tab:red"
    ax.fill_between(x, l, h, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, c, lw=0.9 if not compact else 0.7, color="black")
    if compact:
        ax.set_title(f"{sym[:-4]} {chg:+.1f}%", fontsize=7, pad=2, color=color)
        ax.set_xticks([])
        ax.tick_params(labelsize=5)
    else:
        ax.set_title(f"{cad} -- last {WINDOWS_DAYS[cad]}d "
                     f"({x[0]:%Y-%m-%d} -> {x[-1]:%Y-%m-%d})   "
                     f"close {c[-1]:.6g}   change {chg:+.2f}%",
                     fontsize=9, color=color)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u50")
    args = ap.parse_args()
    spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
    syms = [a["symbol"] for a in spec["assets"]]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = ROOT / "runs" / "mining" / "plots" / f"{args.universe}_subdaily_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-asset pages
    for i, sym in enumerate(syms):
        fig, axes = plt.subplots(4, 1, figsize=(14, 11))
        for ax, cad in zip(axes, CADENCES):
            panel(ax, sym, cad)
        fig.suptitle(f"{sym} -- sub-daily timeframes (data through chimera end)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"{sym}.png", dpi=100)
        plt.close(fig)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(syms)} asset pages done")

    # contact sheets: all assets small-multiple, one cadence per sheet
    for cad in ["1h", "15m"]:
        n = len(syms)
        rows, cols = 10, 5
        fig, axes = plt.subplots(rows, cols, figsize=(16, 22))
        for j, sym in enumerate(syms):
            panel(axes[j // cols][j % cols], sym, cad, compact=True)
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle(f"{args.universe} -- {cad}, last {WINDOWS_DAYS[cad]} days "
                     f"(green=up, red=down over window)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(out_dir / f"_overview_{cad}.png", dpi=100)
        plt.close(fig)
        print(f"contact sheet {cad} done")

    print(f"CHARTS -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
