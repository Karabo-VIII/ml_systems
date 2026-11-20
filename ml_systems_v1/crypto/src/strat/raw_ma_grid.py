"""src/strat/raw_ma_grid.py -- RAW PRICE + MA overlay, asset GRID per timeframe.

USER 2026-06-18: "I want to see raw price + MA overlay, different timeframes. For different assets, not just BTC."
So: one figure PER TIMEFRAME, a grid of assets, each = raw close price (log-y) with a light MA overlay
(fast / mid / slow). Lets the eye check the asset-AGNOSTIC thesis ("one chart looks like another") and read
trend (price above a rising slow MA) vs chop (price whipsawing a flat MA) per asset, per resolution.

MAs are CALENDAR-ANCHORED (default fast=5d, mid=20d, slow=50d) and converted to bars per TF, so the overlay
means the same thing at every timeframe (a native 10/50/200-bar MA would be far too fast at 1h/15m). Warmup =
slow*2+50 bars before the span so the slow MA is formed at the left edge.

RWYB:
  python -m strat.raw_ma_grid                                          # u10, 1d/4h/1h, EMA, 2020
  python -m strat.raw_ma_grid --tfs 1d,4h,1h,30m --ma_type HMA
  python -m strat.raw_ma_grid --assets BTCUSDT,ETHUSDT,SOLUSDT --span 2020-01-01,2023-01-01
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

from strat.ma_2020_breakdown import _panel              # noqa: E402
from strat.ma_type_upgrade import _MA, MA_TYPES         # noqa: E402

OUT = ROOT.parent / "runs" / "strat" / "plots" / "raw_ma_grid"
OUT.mkdir(parents=True, exist_ok=True)

BPD = {"1d": 1, "4h": 6, "2h": 12, "1h": 24, "30m": 48, "15m": 96}
U10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
       "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
MA_DAYS = (5.0, 20.0, 50.0)   # fast, mid, slow (calendar days)
COL_FAST, COL_MID, COL_SLOW = "#ff7f0e", "#2ca02c", "#d62728"


def _periods(tf):
    bpd = BPD[tf]
    return tuple(max(2, int(round(d * bpd))) for d in MA_DAYS)


def _load(sym, tf, span, warmup):
    try:
        o, h, l, c, ms = _panel(sym, tf)
    except Exception:
        return None
    s = pd.Timestamp(span[0]).value // 10**6
    e = pd.Timestamp(span[1]).value // 10**6
    si = max(0, int(np.searchsorted(ms, s)) - warmup)
    ei = int(np.searchsorted(ms, e))
    c2, ms2 = c[si:ei], ms[si:ei]
    if len(c2) < 50:
        return None
    win = ms2 >= s
    if win.sum() < 20:
        return None
    return c2, pd.to_datetime(ms2, unit="ms"), win


def chart(tf, assets, ma_type, span, month_interval):
    p_fast, p_mid, p_slow = _periods(tf)
    warm = p_slow * 2 + 50
    maf = _MA[ma_type]
    overlays = [(f"{ma_type}{p_fast} (~{int(MA_DAYS[0])}d)", p_fast, COL_FAST, 0.9),
                (f"{ma_type}{p_mid} (~{int(MA_DAYS[1])}d)", p_mid, COL_MID, 1.1),
                (f"{ma_type}{p_slow} (~{int(MA_DAYS[2])}d)", p_slow, COL_SLOW, 1.3)]
    n = len(assets); ncol = 2; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 3.0 * nrow), squeeze=False)
    for k, sym in enumerate(assets):
        ax = axes[k // ncol][k % ncol]
        d = _load(sym, tf, span, warm)
        if d is None:
            ax.text(0.5, 0.5, f"{sym}: no/short data", ha="center", va="center")
            ax.set_xticks([]); continue
        c2, idx, win = d
        ax.plot(idx[win], c2[win], color="#1f77b4", lw=1.0, label="close", zorder=3)
        for name, p, col, lw in overlays:
            ma = maf(c2, p)
            ax.plot(idx[win], ma[win], color=col, lw=lw, alpha=0.85, label=name, zorder=4)
        ax.set_yscale("log")
        ax.set_title(f"{sym} @ {tf}", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        if k == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.grid(alpha=0.25)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"RAW PRICE @ {tf} (close, log-y) + {ma_type} overlay "
                 f"[{p_fast}/{p_mid}/{p_slow} bars = ~5d/20d/50d]  span {span[0]}..{span[1]}\n"
                 f"Read: price above a RISING slow MA = trend; price whipsawing a FLAT MA = chop. "
                 f"Asset-agnostic check: does one chart look like another?  Descriptive only.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / f"raw_ma_grid_{tf}_{ma_type}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"   [chart] {p}")
    return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.raw_ma_grid")
    ap.add_argument("--assets", default=",".join(U10))
    ap.add_argument("--tfs", default="1d,4h,1h")
    ap.add_argument("--ma_type", default="EMA", choices=MA_TYPES)
    ap.add_argument("--span", default="2020-01-01,2021-01-01")
    a = ap.parse_args(argv)
    assets = [s.strip() for s in a.assets.split(",") if s.strip()]
    span = tuple(s.strip() for s in a.span.split(","))
    months = (pd.Timestamp(span[1]) - pd.Timestamp(span[0])).days
    month_interval = 1 if months <= 400 else 3
    print(f"## RAW PRICE + {a.ma_type} overlay grid -- {len(assets)} assets, tfs={a.tfs}, span={span}")
    paths = []
    for tf in [t.strip() for t in a.tfs.split(",") if t.strip()]:
        paths.append(chart(tf, assets, a.ma_type, span, month_interval))
    print(f"\n[out] {OUT}")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
