"""src/strat/trade_chart.py -- can a strat TRADE THE CHART? Visual: price + MA cross + IN/OUT + entries/exits.

User /orc 2026-06-13: "not a regime POV -- whether a strat can TRADE A CHART: does it ENTER when there is
opportunity and STAY OUT when needed. Show me ALL the per-instrument x timeframe x MA charts so I can judge
visually." So this is the BEHAVIORAL/VISUAL view (not aggregate %): for each (instrument, timeframe, MA type)
it plots the 2020-H2 price (Jul-Dec, VAL+OOS) with:
  - the fast/slow MA lines (a fixed representative 2MA cross = 50/100, so ONLY the MA TYPE differs);
  - GREEN shading where the FULL-stack strategy is LONG (in position), white where FLAT (out);
  - entry (^) and exit (v) markers;
  - a thin EQUITY curve (right axis) -- did the trading add value;
  - the VAL|OOS split line.
One figure per (instrument, timeframe) = 8 MA-type panels, so you compare all MAs trading the SAME chart.
Saved browsable: runs/periods/TRAIN/2020/MA_2020_BREAKDOWN/charts/trade_charts/<INSTRUMENT>/<TF>.png.
RWYB: python -m strat.trade_chart [--instruments BTCUSDT,..] [--cadences 1d,4h,..]. No emoji (cp1252).
"""
from __future__ import annotations

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

from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA
from strat.ma_2020_breakdown import _panel

MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]
FAST, SLOW = 50, 100                       # fixed representative 2MA cross -> isolates the MA TYPE
WIN = ("2020-07-01", "2021-01-01")         # 2020 H2 = VAL(Jul-Sep) + OOS(Oct-Dec)
SPLIT = "2020-10-01"                        # VAL | OOS
WARMUP_DAYS = 60
ALL_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "MA_2020_BREAKDOWN" / "charts" / "trade_charts"


def _full(c, ma_type):
    """FULL-stack held + position + equity for the 50/100 cross of one MA type."""
    f = _MA[ma_type](c, FAST); s = _MA[ma_type](c, SLOW)
    held = np.nan_to_num(f > s).astype(np.int8)
    held = apply_trail_stop(held.copy(), c, 0.10)[0].astype(np.int8)
    held = min_hold(held, 12).astype(np.int8)
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    pos = np.zeros(len(c)); pos[1:] = held[:-1]
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    eq = np.cumprod(1 + pos * ret - flips * (MAKER_RT / 2.0))
    return f, s, pos, eq


def chart_one(sym, cad):
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return False
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    warm0 = (pd.Timestamp(WIN[0]) - pd.Timedelta(days=WARMUP_DAYS)).value // 10**6
    e = int(np.searchsorted(ms, w1)); s0 = max(0, int(np.searchsorted(ms, warm0)))
    c2, ms2 = c[s0:e], ms[s0:e]
    win = ms2 >= w0
    if win.sum() < 20:                       # not enough 2020-H2 data (e.g. SOL/AVAX new listings)
        return False
    dt = pd.to_datetime(ms2, unit="ms")
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    for ax, mt in zip(axes.ravel(), MA_TYPES):
        f, s, pos, eq = _full(c2, mt)
        px = c2[win]; d = dt[win]; p = pos[win]; eqw = eq[win] / eq[win][0]
        ax.plot(d, px, color="black", lw=0.8, label="price")
        ax.plot(d, f[win], color="#1f77b4", lw=0.7, alpha=0.7, label=f"MA{FAST}")
        ax.plot(d, s[win], color="#d62728", lw=0.7, alpha=0.7, label=f"MA{SLOW}")
        # GREEN shading where LONG (in position)
        ax.fill_between(d, px.min(), px.max(), where=p > 0.5, color="green", alpha=0.13, step="post")
        # entries/exits
        chg = np.diff(np.concatenate([[0.0], p]))
        ent = chg > 0.5; ext = chg < -0.5
        ax.scatter(d[ent], px[ent], marker="^", color="green", s=22, zorder=5)
        ax.scatter(d[ext], px[ext], marker="v", color="red", s=22, zorder=5)
        # OOS compound of THIS config
        oos_mask = d >= pd.Timestamp(SPLIT)
        retw = np.zeros(len(px)); retw[1:] = px[1:] / px[:-1] - 1
        posw = p.copy()
        oo = float(np.prod(1 + (posw * retw)[oos_mask]) - 1) * 100 if oos_mask.sum() > 3 else float("nan")
        ax.axvline(pd.Timestamp(SPLIT), color="grey", ls="--", lw=0.8)
        ax.set_title(f"{mt}  (50/100 cross)  OOS {oo:+.0f}%  | green=LONG", fontsize=9)
        ax.tick_params(labelsize=7)
        ax2 = ax.twinx(); ax2.plot(d, eqw, color="purple", lw=1.0, alpha=0.6)
        ax2.tick_params(labelsize=6); ax2.set_ylabel("eq", fontsize=6, color="purple")
        if mt == "EMA":
            ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(f"{sym} @ {cad} -- 2020 H2 (VAL Jul-Sep | OOS Oct-Dec): does each MA TRADE THE CHART? "
                 f"(green=in position; ^=entry v=exit; purple=equity)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    od = OUT / sym; od.mkdir(parents=True, exist_ok=True)
    fig.savefig(od / f"{cad}.png", dpi=95); plt.close(fig)
    return True


def main() -> int:
    syms = ALL_SYMS; cads = ["1d", "4h", "2h", "1h", "30m", "15m"]
    if "--instruments" in sys.argv:
        syms = sys.argv[sys.argv.index("--instruments") + 1].split(",")
    if "--cadences" in sys.argv:
        cads = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    print(f"trade charts: {syms} x {cads}; cross={FAST}/{SLOW}; window {WIN}")
    made = 0
    for sym in syms:
        for cad in cads:
            ok = chart_one(sym, cad)
            print(f"  {sym:10} {cad:4} {'OK' if ok else 'skip (no H2 data)'}", flush=True)
            made += int(ok)
    print(f"\n{made} charts under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
