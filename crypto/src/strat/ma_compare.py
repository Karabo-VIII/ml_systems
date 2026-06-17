"""src/strat/ma_compare.py -- old (baseline) vs new (min-hold fix) + residual weaknesses + family consistency.

WHY (user /orc 2026-06-12): "compare results side by side: old vs new for the SAME strats/timeframes
where we observed weaknesses. Any OTHER weakness? And which strat FAMILY showed sustained + consistent
performance over the month?" Three sections:
  A. OLD vs NEW on the WEAK cells (fast/mid MA at 30m/15m) -- baseline net vs min_hold(12) net.
  B. RESIDUAL / OTHER weaknesses -- what STILL loses after the fix, where the fix HURTS, per-asset gaps,
     and the NEW weakness the min-hold introduces (held-through-reversal -> deeper drawdown?).
  C. FAMILY CONSISTENCY -- 2MA vs 3MA x speed bucket: mean net, breadth (%positive), and SUSTAINED
     (avg max-DD + % of bars in profit). "Sustained+consistent" = high %pos + decent mean + shallow DD.

Recomputes a config sample (40 distinct 2MA + 40 3MA, spread fast->slow) x 7 assets x {baseline,
min_hold12} x 4 cadences with rich per-cell metrics (net, maxDD, %-bars-in-profit, trades). Oldest
month (TRAIN/2020/JAN), taker. Stores chart in the period folder. RWYB: python -m strat.ma_compare
No emoji (cp1252).
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, TAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.period_store import sub as period_sub

TFS = ["4h", "1h", "30m", "15m"]
START, END = "2020-01-07", "2020-02-07"
SPEED = lambda cfg: (lambda n: ("fast(<20)" if max(n) < 20 else "mid(20-60)" if max(n) < 60
                                else "slow(60-150)" if max(n) < 150 else "vslow(>=150)"))([int(x) for x in re.findall(r"\d+", cfg)])
BUCKETS = ["fast(<20)", "mid(20-60)", "slow(60-150)", "vslow(>=150)"]


def rich(held, c, cost=TAKER_RT):
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    pos = np.zeros(len(c)); pos[1:] = held[:-1]
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    eq = np.cumprod(1 + pos * ret - flips * (cost / 2.0))
    peak = np.maximum.accumulate(eq)
    maxdd = float((((eq - peak) / peak)).min() * 100)
    return (float(eq[-1] - 1) * 100, maxdd, float((eq > 1).mean()) * 100,
            int((np.diff(np.concatenate([[0], held])) == 1).sum()))


def run(n_cfg=40):
    s_ms = pd.Timestamp(START).value // 10**6; e_ms = pd.Timestamp(END).value // 10**6
    specs = {}
    fam_of = {}
    for fam in ("2MA", "3MA"):
        s = distinct_specs(fam, 0.15, max_n=n_cfg); specs.update(s)
        for k in s: fam_of[k] = fam
    PR.STRATS.update(specs)
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    cells = []
    for cad in TFS:
        for sym in syms:
            try:
                o, h, l, c, ms = _cached_panel(sym, cad)
            except Exception:
                continue
            keep = ms < e_ms; o, c, ms = o[keep], c[keep], ms[keep]
            if (ms >= s_ms).sum() < 5:
                continue
            for name in specs:
                h0 = holding_state(name, o, c, c, c).astype(np.int8)
                for ov, held in (("base", h0), ("minhold", min_hold(h0, 12).astype(np.int8))):
                    net, dd, fip, ntr = rich(held, c)
                    cells.append({"cadence": cad, "asset": sym, "config": name, "family": fam_of[name],
                                  "speed": SPEED(name), "overlay": ov, "net": net, "maxdd": dd,
                                  "fip": fip, "trades": ntr})
    return cells


def _agg(cells, pred):
    v = [c for c in cells if pred(c)]
    if not v: return None
    return (np.mean([c["net"] for c in v]), 100 * np.mean([c["net"] for c in v]) and 100 * np.mean(np.array([c["net"] for c in v]) > 0),
            np.mean([c["maxdd"] for c in v]), np.mean([c["fip"] for c in v]), len(v))


def main(argv=None) -> int:
    global START, END
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START); ap.add_argument("--end", default=END)
    a = ap.parse_args(argv); START, END = a.start, a.end
    print(f"(window {START}..{END})")
    cells = run()
    base = {(c["cadence"], c["config"], c["asset"]): c for c in cells if c["overlay"] == "base"}
    # ---------- A. OLD vs NEW on the WEAK cells (fast/mid at 30m/15m) ----------
    print("## [A] OLD (baseline) vs NEW (min-hold 12) -- the WEAK cells (fast/mid MA at 30m & 15m)\n")
    print(f"   {'cadence':8} {'speed':12} {'base net%':>10} {'minhold%':>10} {'delta':>8} {'base %pos':>10} {'new %pos':>9}")
    for cad in ["30m", "15m"]:
        for sp in ["fast(<20)", "mid(20-60)"]:
            b = np.array([c["net"] for c in cells if c["overlay"] == "base" and c["cadence"] == cad and c["speed"] == sp])
            m = np.array([c["net"] for c in cells if c["overlay"] == "minhold" and c["cadence"] == cad and c["speed"] == sp])
            if not len(b): continue
            print(f"   {cad:8} {sp:12} {b.mean():>10.1f} {m.mean():>10.1f} {m.mean()-b.mean():>+8.1f} "
                  f"{100*(b>0).mean():>9.0f}% {100*(m>0).mean():>8.0f}%")
    # ---------- B. RESIDUAL / OTHER weaknesses ----------
    print("\n## [B] RESIDUAL / OTHER weaknesses\n")
    # B1 -- still-negative after the fix
    print("   B1. STILL net-negative after min-hold (by cadence x speed):")
    for cad in TFS:
        for sp in BUCKETS:
            m = np.array([c["net"] for c in cells if c["overlay"] == "minhold" and c["cadence"] == cad and c["speed"] == sp])
            if len(m) and m.mean() < 0:
                print(f"      {cad:5} {sp:12} still {m.mean():6.1f}% ({100*(m>0).mean():3.0f}% pos)")
    # B2 -- where the fix HURTS (minhold < base)
    print("   B2. where min-hold HURTS (delta<0):")
    hurt = False
    for cad in TFS:
        for sp in BUCKETS:
            b = np.mean([c["net"] for c in cells if c["overlay"] == "base" and c["cadence"] == cad and c["speed"] == sp] or [0])
            m = np.mean([c["net"] for c in cells if c["overlay"] == "minhold" and c["cadence"] == cad and c["speed"] == sp] or [0])
            if m < b - 0.3:
                print(f"      {cad:5} {sp:12} {b:6.1f}% -> {m:6.1f}% (delta {m-b:+.1f})"); hurt = True
    if not hurt: print("      (none -- min-hold did not hurt any cadence x speed bucket)")
    # B3 -- the NEW weakness: drawdown cost of holding through reversals
    bdd = np.mean([c["maxdd"] for c in cells if c["overlay"] == "base"])
    mdd = np.mean([c["maxdd"] for c in cells if c["overlay"] == "minhold"])
    print(f"   B3. drawdown side-effect of min-hold: avg maxDD base {bdd:.1f}% -> minhold {mdd:.1f}%  "
          f"({'DEEPER (holds through reversals)' if mdd < bdd else 'no worse'})")
    # B4 -- per-asset residual
    print("   B4. per-asset (minhold mean net, %pos) -- the lagging instrument:")
    for a in sorted({c["asset"] for c in cells}):
        m = np.array([c["net"] for c in cells if c["overlay"] == "minhold" and c["asset"] == a])
        print(f"      {a.replace('USDT',''):5} {m.mean():6.1f}% ({100*(m>0).mean():3.0f}% pos)")

    # ---------- C. FAMILY CONSISTENCY ----------
    print("\n## [C] FAMILY CONSISTENCY over the month (baseline; sustained = shallow DD + bars-in-profit)\n")
    print(f"   {'family':6} {'speed':12} {'mean net%':>10} {'std':>6} {'%pos':>5} {'avg maxDD%':>11} {'%bars profit':>13}")
    rows = []
    for fam in ("2MA", "3MA"):
        for sp in BUCKETS:
            v = [c for c in cells if c["overlay"] == "base" and c["family"] == fam and c["speed"] == sp]
            if not v: continue
            nets = np.array([c["net"] for c in v])
            r = (fam, sp, nets.mean(), nets.std(), 100*(nets>0).mean(),
                 np.mean([c["maxdd"] for c in v]), np.mean([c["fip"] for c in v]))
            rows.append(r)
            print(f"   {fam:6} {sp:12} {r[2]:>10.1f} {r[3]:>6.1f} {r[4]:>4.0f}% {r[5]:>10.1f}% {r[6]:>12.0f}%")
    # the most sustained+consistent (high %pos + shallow DD + decent mean)
    best = max(rows, key=lambda r: r[4] - abs(r[5]) * 0.5 + r[2] * 0.3)
    print(f"\n   => MOST SUSTAINED+CONSISTENT: {best[0]} {best[1]}  "
          f"(mean {best[2]:.1f}%, {best[4]:.0f}% pos, maxDD {best[5]:.1f}%, {best[6]:.0f}% bars in profit)")

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    # A: old vs new on weak cells
    labels, bvals, mvals = [], [], []
    for cad in ["30m", "15m"]:
        for sp in ["fast(<20)", "mid(20-60)"]:
            b = np.mean([c["net"] for c in cells if c["overlay"]=="base" and c["cadence"]==cad and c["speed"]==sp] or [np.nan])
            m = np.mean([c["net"] for c in cells if c["overlay"]=="minhold" and c["cadence"]==cad and c["speed"]==sp] or [np.nan])
            labels.append(f"{cad}\n{sp.split('(')[0]}"); bvals.append(b); mvals.append(m)
    x = np.arange(len(labels)); w = 0.38
    ax[0].bar(x - w/2, bvals, w, label="OLD (baseline)", color="#cb181d")
    ax[0].bar(x + w/2, mvals, w, label="NEW (min-hold)", color="#2c7fb8")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8); ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_ylabel("mean net %"); ax[0].legend(); ax[0].set_title("[A] OLD vs NEW on the weak cells (fast/mid @ 30m,15m)")
    # C: family consistency -- %pos vs maxDD scatter
    for fam, col in (("2MA", "#1f77b4"), ("3MA", "#ff7f0e")):
        for sp in BUCKETS:
            v = [c for c in cells if c["overlay"]=="base" and c["family"]==fam and c["speed"]==sp]
            if not v: continue
            pos = 100*np.mean(np.array([c["net"] for c in v])>0); dd = np.mean([c["maxdd"] for c in v])
            ax[1].scatter(dd, pos, s=90, color=col, alpha=0.8)
            ax[1].annotate(f"{fam[:1]}{sp[:1]}", (dd, pos), fontsize=7)
    ax[1].set_xlabel("avg max drawdown % (left = sustained)"); ax[1].set_ylabel("% configs positive (up = consistent)")
    ax[1].set_title("[C] family consistency: up-and-left = sustained+consistent\n(blue=2MA, orange=3MA)")
    fig.tight_layout()
    out = period_sub(START, "charts") / "ma_compare.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
