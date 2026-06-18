"""src/strat/ma_slice_charts.py -- 2MA & 3MA overlaid on RAW PRICE, per 2-week slice, per timeframe.

USER 2026-06-18: "Draw some 2MA and 3MA on them at different timeframes. Pick 2-week slices of each.
Run for 4 different 2-week slices." -- so the user can SEE the move-catch behaviour (catch / lag / whipsaw)
of moving-average crosses across regimes and resolutions, on the actual candles.

For each (slice, timeframe) the figure has TWO price panels:
  - TOP    = candles + 2MA (fast/slow cross); long-region (fast>slow) shaded green; entry ^ / exit v marked.
  - BOTTOM = candles + 3MA (fast/mid/slow stack); long-region (fast>mid>slow) shaded; entry/exit marked.
MAs are CALENDAR-ANCHORED (same calendar speed across TFs) so the cross behaviour is comparable across
resolutions: 2MA = (1d, 4d); 3MA = (1d, 3d, 8d), converted to bars per TF. MA type configurable (default EMA).

Warmup: MAs are computed on history BEFORE the slice (slow*3 + 60 bars) then only the slice window is drawn,
so the slow MA is fully formed at the left edge.

RWYB:
  python -m strat.ma_slice_charts                              # BTC, 4 default slices, 4h/1h/15m, EMA
  python -m strat.ma_slice_charts --asset ETHUSDT --ma_type HMA --tfs 4h,1h
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations

import argparse
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

from strat.ma_2020_breakdown import _panel              # noqa: E402
from strat.ma_type_upgrade import _MA, MA_TYPES         # noqa: E402

OUT = ROOT.parent / "runs" / "strat" / "plots" / "ma_slices_2020"
OUT.mkdir(parents=True, exist_ok=True)

BPD = {"1d": 1, "4h": 6, "2h": 12, "1h": 24, "30m": 48, "15m": 96}

# Four 2-week slices spanning 2020's move archetypes
SLICES = [
    ("crash",     "2020-03-09", "2020-03-23", "COVID CRASH (violent down + V-bottom)"),
    ("recovery",  "2020-04-22", "2020-05-06", "RECOVERY TREND (clean grind up)"),
    ("chop",      "2020-09-02", "2020-09-16", "CHOP / RANGE (Sept pullback, whipsaw)"),
    ("breakout",  "2020-11-01", "2020-11-15", "Q4 BULL BREAKOUT (strong trend leg)"),
]

# Calendar-anchored MA spans (days). Converted to bars per TF.
MA2_DAYS = (1.0, 4.0)         # fast, slow
MA3_DAYS = (1.0, 3.0, 8.0)    # fast, mid, slow

COL_UP, COL_DN = "#26a69a", "#ef5350"
COL_FAST, COL_MID, COL_SLOW = "#1f77b4", "#9467bd", "#d62728"


def _periods(tf, days_tuple):
    bpd = BPD[tf]
    return tuple(max(2, int(round(d * bpd))) for d in days_tuple)


def _load_slice(sym, tf, start, end, warmup_bars):
    o, h, l, c, ms = _panel(sym, tf)
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    s_idx = max(0, int(np.searchsorted(ms, s_ms)) - warmup_bars)
    e_idx = int(np.searchsorted(ms, e_ms))
    if e_idx - s_idx < 10:
        return None
    o2, h2, l2, c2, ms2 = o[s_idx:e_idx], h[s_idx:e_idx], l[s_idx:e_idx], c[s_idx:e_idx], ms[s_idx:e_idx]
    win = ms2 >= s_ms
    return o2, h2, l2, c2, ms2, win


def _candles(ax, o, h, l, c, x):
    up = c >= o
    for col, m in [(COL_UP, up), (COL_DN, ~up)]:
        if not m.any():
            continue
        ax.vlines(x[m], l[m], h[m], color=col, lw=0.6, alpha=0.9, zorder=2)
        ax.vlines(x[m], np.minimum(o, c)[m], np.maximum(o, c)[m], color=col, lw=2.4, alpha=0.95, zorder=3)


def _close_line(ax, c, x):
    ax.plot(x, c, color="#455a64", lw=0.9, zorder=2)


def _shade_long(ax, sig, x):
    """Shade contiguous long runs (green) and mark entry ^ / exit v."""
    sig = sig.astype(int)
    d = np.diff(np.concatenate([[0], sig, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        ax.axvspan(x[s] - 0.5, x[min(e, len(x) - 1)] - 0.5, color="#26a69a", alpha=0.10, zorder=1)


def _markers(ax, sig, c, x):
    d = np.diff(np.concatenate([[0], sig.astype(int), [0]]))
    ent = np.where(d == 1)[0]
    ext = np.where(d == -1)[0]
    ent = ent[ent < len(x)]
    ext = ext[ext < len(x)]
    if len(ent):
        ax.scatter(x[ent], c[ent], marker="^", s=70, color="#1b5e20", edgecolor="white",
                   linewidth=0.6, zorder=6, label="entry")
    if len(ext):
        ax.scatter(x[ext], c[ext], marker="v", s=70, color="#b71c1c", edgecolor="white",
                   linewidth=0.6, zorder=6, label="exit")


def _xticks(ax, idx_win, x):
    n = len(x)
    k = max(1, n // 8)
    ticks = list(range(0, n, k))
    ax.set_xticks([x[t] for t in ticks])
    labels = [pd.Timestamp(idx_win[t]).strftime("%m-%d\n%H:%M") for t in ticks]
    ax.set_xticklabels(labels, fontsize=7)


def slice_figure(sym, tf_list, slc, ma_type):
    name, start, end, desc = slc
    nrow = len(tf_list)
    fig, axes = plt.subplots(nrow, 2, figsize=(19, 4.3 * nrow), squeeze=False)
    maf = _MA[ma_type]
    for r, tf in enumerate(tf_list):
        p2 = _periods(tf, MA2_DAYS)
        p3 = _periods(tf, MA3_DAYS)
        warm = max(p2[-1], p3[-1]) * 3 + 60
        d = _load_slice(sym, tf, start, end, warm)
        for col in (0, 1):
            ax = axes[r][col]
            if d is None:
                ax.text(0.5, 0.5, f"{sym} {tf}: no data", ha="center", va="center")
                ax.set_xticks([])
                continue
            o, h, l, c, ms, win = d
            idx = pd.to_datetime(ms, unit="ms")
            wi = np.where(win)[0]
            ow, hw, lw_, cw = o[wi], h[wi], l[wi], c[wi]
            xw = np.arange(len(wi))
            idx_win = idx[wi]
            dense = len(wi) > 500
            if dense:
                _close_line(ax, cw, xw)
            else:
                _candles(ax, ow, hw, lw_, cw, xw)
            if col == 0:  # 2MA
                fma = maf(c, p2[0]); sma = maf(c, p2[1])
                fw, sw = fma[wi], sma[wi]
                ax.plot(xw, fw, color=COL_FAST, lw=1.3, label=f"{ma_type}{p2[0]} (fast~1d)", zorder=4)
                ax.plot(xw, sw, color=COL_SLOW, lw=1.5, label=f"{ma_type}{p2[1]} (slow~4d)", zorder=4)
                sig = np.nan_to_num(fw > sw)
                ttl = f"2MA cross  {ma_type}({p2[0]},{p2[1]})"
            else:        # 3MA
                fma = maf(c, p3[0]); mma = maf(c, p3[1]); sma = maf(c, p3[2])
                fw, mw, sw = fma[wi], mma[wi], sma[wi]
                ax.plot(xw, fw, color=COL_FAST, lw=1.2, label=f"{ma_type}{p3[0]} (fast~1d)", zorder=4)
                ax.plot(xw, mw, color=COL_MID, lw=1.2, label=f"{ma_type}{p3[1]} (mid~3d)", zorder=4)
                ax.plot(xw, sw, color=COL_SLOW, lw=1.5, label=f"{ma_type}{p3[2]} (slow~8d)", zorder=4)
                sig = np.nan_to_num((fw > mw) & (mw > sw))
                ttl = f"3MA stack  {ma_type}({p3[0]},{p3[1]},{p3[2]})"
            _shade_long(ax, sig, xw)
            _markers(ax, sig, cw, xw)
            _xticks(ax, idx_win, xw)
            cov = 100.0 * sig.mean() if len(sig) else 0.0
            ax.set_title(f"{tf}  |  {ttl}   [long {cov:.0f}% of window]", fontsize=10)
            ax.grid(alpha=0.20)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    fig.suptitle(f"{sym}  --  {name.upper()}  {start} -> {end}  ({desc})\n"
                 f"green shade = MA says LONG;  ^ entry  v exit;  {ma_type} MAs, calendar-anchored "
                 f"(same speed across TFs).  Dense TFs drawn as close-line.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / f"maslice_{sym}_{name}_{ma_type}.png"
    fig.savefig(p, dpi=115); plt.close(fig)
    print(f"   [chart] {p}")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_slice_charts")
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--tfs", default="4h,1h,15m")
    ap.add_argument("--ma_type", default="EMA", choices=MA_TYPES)
    a = ap.parse_args(argv)
    tf_list = [t.strip() for t in a.tfs.split(",") if t.strip()]
    print(f"## MA SLICE charts -- {a.asset}, ma_type={a.ma_type}, tfs={tf_list}, {len(SLICES)} slices")
    paths = []
    for slc in SLICES:
        paths.append(str(slice_figure(a.asset, tf_list, slc, a.ma_type)))
    print(f"\n[out] {OUT}")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
