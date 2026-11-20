"""src/strat/structural_fixes.py -- mine STRUCTURAL solutions to the per-timeframe MA failures.

WHY (user /orc 2026-06-12): "identify potential structural gaps, solve them early, then see if we get
better performance ... mine potential structural solution to failures per timeframe." The killer
(ma_killers) is over-trading via MA-speed/cadence MISMATCH -> cost drag; whipsaw is ~10-16% of it. So
this tests STRUCTURAL OVERLAYS on the holding state -- the same MA signal, but with a discipline layer:
  - baseline      : the raw MA signal-flip (no overlay)
  - cooldown(N)   : after an EXIT, block re-entry for N bars (the user's idea; attacks whipsaw)
  - min_hold(M)   : after an ENTRY, hold >= M bars before any exit (forces the move to develop)
  - confirm(K)    : only ENTER if the signal has been true K consecutive bars (debounce the cross)
  - cooldown+confirm : the combination
Each overlay is applied to held, then the bar-level MtM net (causal, taker cost) is recomputed. We
report, PER CADENCE: mean net, % positive, trades, whipsaw -- and whether the overlay LIFTS the failing
cadences (30m/15m) WITHOUT hurting 4h. The winner per cadence is the structural fix to carry forward.

Config sample: 16 distinct 2MA + 16 distinct 3MA per family (evenly spread fast->slow) x 7 assets.
DESCRIPTIVE / TRAIN-period (oldest month). Stores outputs in the period folder via period_store.
RWYB: python -m strat.structural_fixes
No emoji (cp1252).
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

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, TAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.period_store import sub as period_sub

TFS = ["4h", "1h", "30m", "15m"]
START, END = "2020-01-07", "2020-02-07"
CAD_COLOR = {"4h": "#1b9e77", "1h": "#7570b3", "30m": "#d95f02", "15m": "#e7298a"}


# ---- structural overlays on the binary held series -------------------------------------------
def cooldown(held, n):
    out = held.copy(); block = 0
    for i in range(1, len(out)):
        if block > 0:
            out[i] = 0; block -= 1
        if held[i - 1] == 1 and held[i] == 0:        # an exit just happened
            block = n
    return out


def min_hold(held, m):
    out = held.copy(); hold_left = 0
    for i in range(1, len(out)):
        if out[i - 1] == 0 and held[i] == 1:         # entry
            hold_left = m
        if hold_left > 0:
            out[i] = 1; hold_left -= 1
        else:
            out[i] = held[i]
    return out


def confirm(held, k):
    out = np.zeros_like(held)
    run = 0
    for i in range(len(held)):
        run = run + 1 if held[i] == 1 else 0
        out[i] = 1 if run >= k else 0
    return out


OVERLAYS = {
    "baseline": lambda h: h,
    "cooldown6": lambda h: cooldown(h, 6),
    "min_hold12": lambda h: min_hold(h, 12),
    "confirm3": lambda h: confirm(h, 3),
    "cool6+conf3": lambda h: cooldown(confirm(h, 3), 6),
}


def _net(held, c, cost=TAKER_RT):
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    pos = np.zeros(len(c)); pos[1:] = held[:-1]
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net_bar = pos * ret - flips * (cost / 2.0)
    n_tr = int((np.diff(np.concatenate([[0], held])) == 1).sum())
    # whipsaw = entries whose hold <=2 bars
    whip = 0; i = 0
    while i < len(held):
        if held[i] == 1 and (i == 0 or held[i - 1] == 0):
            j = i
            while j < len(held) and held[j] == 1: j += 1
            if (j - i) <= 2: whip += 1
            i = j
        else:
            i += 1
    return float(np.cumprod(1 + net_bar)[-1] - 1) * 100, n_tr, whip


def run():
    s_ms = pd.Timestamp(START).value // 10**6
    e_ms = pd.Timestamp(END).value // 10**6
    specs = {}
    for fam in ("2MA", "3MA"):
        specs.update(distinct_specs(fam, 0.15, max_n=16))
    PR.STRATS.update(specs)
    import yaml
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    cells = []
    for cad in TFS:
        for sym in syms:
            try:
                o, h, l, c, ms = _cached_panel(sym, cad)
            except Exception:
                continue
            keep = ms < e_ms
            o, c, ms = o[keep], c[keep], ms[keep]
            wm = ms >= s_ms
            if wm.sum() < 5:
                continue
            for name in specs:
                held0 = holding_state(name, o, c, c, c).astype(np.int8)
                for ov, fn in OVERLAYS.items():
                    held = fn(held0).astype(np.int8)
                    net, ntr, whip = _net(held, c)
                    cells.append({"cadence": cad, "asset": sym, "config": name, "overlay": ov,
                                  "net": net, "trades": ntr, "whip": whip})
    return cells


def report(cells):
    print(f"## STRUCTURAL FIXES per timeframe -- {len(cells)} cells (32 configs x 7 assets x {len(OVERLAYS)} overlays)\n")
    print(f"   {'cadence':8} {'overlay':14} {'mean net%':>10} {'%pos':>6} {'trades':>7} {'whip':>6} {'vs base':>8}")
    summary = {}
    for cad in TFS:
        base = np.mean([x["net"] for x in cells if x["cadence"] == cad and x["overlay"] == "baseline"])
        for ov in OVERLAYS:
            v = np.array([x["net"] for x in cells if x["cadence"] == cad and x["overlay"] == ov])
            tr = np.mean([x["trades"] for x in cells if x["cadence"] == cad and x["overlay"] == ov])
            wh = np.mean([x["whip"] for x in cells if x["cadence"] == cad and x["overlay"] == ov])
            summary[(cad, ov)] = (v.mean(), 100 * (v > 0).mean(), tr, wh, v.mean() - base)
        best_delta = max(summary[(cad, o2)][4] for o2 in OVERLAYS if o2 != "baseline")
        for ov in OVERLAYS:
            m, pos, tr, wh, delta = summary[(cad, ov)]
            star = " <==" if ov != "baseline" and delta == best_delta else ""
            print(f"   {cad:8} {ov:14} {m:>10.1f} {pos:>5.0f}% {tr:>7.0f} {wh:>6.1f} {delta:>+8.1f}{star}")
        print()
    return summary


def figure(cells, summary, out):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ovs = list(OVERLAYS)
    x = np.arange(len(TFS)); w = 0.16
    for i, ov in enumerate(ovs):
        vals = [summary[(cad, ov)][0] for cad in TFS]
        ax[0].bar(x + (i - 2) * w, vals, w, label=ov)
    ax[0].set_xticks(x); ax[0].set_xticklabels(TFS); ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_ylabel("mean net %"); ax[0].legend(fontsize=8)
    ax[0].set_title("Structural overlay x cadence -- mean net\n(does the overlay lift the failing cadences?)")
    # delta vs baseline
    for i, ov in enumerate([o for o in ovs if o != "baseline"]):
        deltas = [summary[(cad, ov)][4] for cad in TFS]
        ax[1].plot(TFS, deltas, "o-", label=ov, lw=2)
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_ylabel("net delta vs baseline (pp)")
    ax[1].legend(fontsize=8); ax[1].set_title("Net improvement vs baseline, per cadence\n(>0 = the fix helps)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main(argv=None) -> int:
    global START, END
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START); ap.add_argument("--end", default=END)
    a = ap.parse_args(argv); START, END = a.start, a.end
    print(f"(window {START}..{END})")
    cells = run()
    summary = report(cells)
    out = period_sub(START, "charts") / "structural_fixes.png"
    figure(cells, summary, out)
    print(f"[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
