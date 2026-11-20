"""src/strat/raw_price_charts.py -- RAW PRICE charts (the timeframe itself) with a FEW MA overlays.

USER /orc 2026-06-16: "I want to see the RAW timeframe charts, not just the MA results -- publish those with
configs layered on top (not a lot), so I can see if we have trends or not." So: actual close-price per asset per
TF, with a light set of MAs overlaid (fast / mid / slow), across the full 2020->2022 arc (bull -> mixed -> bear)
with year boundaries marked, so trends vs chop are VISIBLE. Long-only context; descriptive only (no signal/PnL).

Overlays (deliberately few -- "not a lot"): EMA(10) fast, EMA(50) mid, SMA(200) slow. These let the eye read
trend (price above rising slow MA) vs chop (price whipsawing around a flat MA) directly.

RWYB: python -m strat.raw_price_charts --tfs 1d,4h --assets BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations

import argparse
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

from strat.ma_2020_breakdown import _panel                                   # noqa: E402
from strat.ma_type_upgrade import _ema, _sma                                 # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "RAW_PRICE"
OUT.mkdir(parents=True, exist_ok=True)
SPAN = ("2020-01-01", "2023-01-01")                                          # the full bull->mixed->bear arc
YEAR_BOUNDS = ["2021-01-01", "2022-01-01"]
OVERLAYS = [("EMA(10)", lambda c: _ema(c, 10), "#ff7f0e", 0.9),
            ("EMA(50)", lambda c: _ema(c, 50), "#2ca02c", 1.1),
            ("SMA(200)", lambda c: _sma(c, 200), "#d62728", 1.3)]


def _load(sym, cad, span):
    o, h, l, c, ms = _panel(sym, cad)
    s = pd.Timestamp(span[0]).value // 10**6; e = pd.Timestamp(span[1]).value // 10**6
    si = max(0, int(np.searchsorted(ms, s)) - 250)                           # warmup for the slow MA
    ei = int(np.searchsorted(ms, e))
    c2, ms2 = c[si:ei], ms[si:ei]
    if len(c2) < 50:
        return None
    idx = pd.to_datetime(ms2, unit="ms")
    win = ms2 >= s
    return c2, idx, win


def chart(tf, assets):
    n = len(assets)
    ncol = 2; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 3.2 * nrow), squeeze=False)
    for k, sym in enumerate(assets):
        ax = axes[k // ncol][k % ncol]
        d = _load(sym, tf, SPAN)
        if d is None:
            ax.text(0.5, 0.5, f"{sym}: no data", ha="center", va="center"); ax.set_xticks([]); continue
        c2, idx, win = d
        ax.plot(idx[win], c2[win], color="#1f77b4", lw=1.0, label="close", zorder=3)
        for name, fn, col, lw in OVERLAYS:
            ma = fn(c2)
            ax.plot(idx[win], ma[win], color=col, lw=lw, alpha=0.85, label=name, zorder=4)
        for yb in YEAR_BOUNDS:
            ax.axvline(pd.Timestamp(yb), color="grey", ls="--", lw=0.8, alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"{sym} @ {tf}", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        if k == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"RAW PRICE @ {tf} (close, log-y) + light MA overlays [EMA10/EMA50/SMA200] -- 2020 bull | 2021 "
                 f"mixed (mega-bull H1 -> May crash -> H2 -> Q4 decline) | 2022 BEAR. Dashed = year boundary.\n"
                 f"Read: price above a RISING slow MA = trend (MA strats work); price whipsawing a FLAT MA = chop "
                 f"(MA strats whipsaw -> need faster MAs / cash). Descriptive only.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / f"raw_price_ma_overlay_{tf}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"   [chart] {p}")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.raw_price_charts")
    ap.add_argument("--tfs", default="1d,4h")
    ap.add_argument("--assets", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT")
    a = ap.parse_args(argv)
    assets = [s.strip() for s in a.assets.split(",") if s.strip()]
    print(f"## RAW PRICE charts (close + EMA10/EMA50/SMA200) -- 2020-2022, {len(assets)} assets")
    for tf in [t.strip() for t in a.tfs.split(",") if t.strip()]:
        chart(tf, assets)
    print(f"\n[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
