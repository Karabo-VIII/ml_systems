"""src/strat/combined_analysis.py -- the COMBINED multi-month scenario: Jan + Feb 2020 as one run.

WHY (user /orc 2026-06-12): "now you have a combined multi-month scenario: analyse the Jan and Feb ROI,
returns, strats, exit mechanisms, etc -- everything; tell me strengths/weaknesses." So this runs the MA
configs CONTINUOUSLY over 2020-01-07 -> 2020-03-07 (equity compounds through the Jan RALLY and the Feb
TOP+reversal/COVID-onset), and decomposes each cell's ROI into its JAN-half and FEB-half so we see who
HELD their gains. The combined span is where EXIT mechanism finally matters (in a pure rally it didn't):
a trailing stop locks Jan's gains before the Feb drop; signal-flip/min-hold may give some back.

Per (config, cadence, EXIT in {signalflip, trail5, trail10, minhold12}) x 7 assets:
  combined ROI, JAN-half ROI, FEB-half ROI, maxDD (the Feb reversal), %bars-in-profit, trades.
Aggregates: ROI by cadence x exit; family (2MA/3MA x speed) combined + Feb-half robustness; the EXIT
ranking over the span; best combined config per cadence. Chart: combined equity of the best config per
cadence (rally->reversal) + exit comparison. Stored in TRAIN/2020/JAN_FEB_COMBINED/.
RWYB: python -m strat.combined_analysis
No emoji (cp1252).
"""
from __future__ import annotations

import re
import sys
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
from strat.portfolio_replay import holding_state, apply_trail_stop, TAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold

START, SPLIT, END = "2020-01-07", "2020-02-07", "2020-03-07"
TFS = ["4h", "1h", "30m", "15m"]
OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "JAN_FEB_COMBINED"
(OUT / "charts").mkdir(parents=True, exist_ok=True); (OUT / "analysis").mkdir(parents=True, exist_ok=True)
SPEED = lambda cfg: (lambda n: ("fast(<20)" if max(n) < 20 else "mid(20-60)" if max(n) < 60
                                else "slow(60-150)" if max(n) < 150 else "vslow(>=150)"))([int(x) for x in re.findall(r"\d+", cfg)])
BUCKETS = ["fast(<20)", "mid(20-60)", "slow(60-150)", "vslow(>=150)"]
EXITS = ["signalflip", "trail5", "trail10", "minhold12"]


def _held(name, o, c, exit_):
    h0 = holding_state(name, o, c, c, c).astype(np.int8)
    if exit_ == "trail5": return apply_trail_stop(h0.copy(), c, 0.05)[0].astype(np.int8)
    if exit_ == "trail10": return apply_trail_stop(h0.copy(), c, 0.10)[0].astype(np.int8)
    if exit_ == "minhold12": return min_hold(h0, 12).astype(np.int8)
    return h0


def _eq(held, c, cost=TAKER_RT):
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    pos = np.zeros(len(c)); pos[1:] = held[:-1]
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return np.cumprod(1 + pos * ret - flips * (cost / 2.0)), int((np.diff(np.concatenate([[0], held])) == 1).sum())


def run(n_cfg=24):
    s_ms = pd.Timestamp(START).value // 10**6; sp_ms = pd.Timestamp(SPLIT).value // 10**6
    e_ms = pd.Timestamp(END).value // 10**6
    specs, fam_of = {}, {}
    for fam in ("2MA", "3MA"):
        s = distinct_specs(fam, 0.15, max_n=n_cfg); specs.update(s)
        for k in s: fam_of[k] = fam
    PR.STRATS.update(specs)
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    cells = []; curves = {}
    for cad in TFS:
        for sym in syms:
            try:
                o, h, l, c, ms = _cached_panel(sym, cad)
            except Exception:
                continue
            keep = ms < e_ms; o, c, ms = o[keep], c[keep], ms[keep]
            wm = ms >= s_ms
            if wm.sum() < 10:
                continue
            spi = int(np.searchsorted(ms[wm], sp_ms))             # split index within window
            for name in specs:
                for ex in EXITS:
                    held = _held(name, o, c, ex)
                    eqf, ntr = _eq(held, c)
                    eq = eqf[wm]
                    if len(eq) <= spi or spi <= 0:
                        continue
                    comb = float(eq[-1] / eq[0] - 1) * 100
                    jan = float(eq[spi] / eq[0] - 1) * 100
                    feb = float(eq[-1] / eq[spi] - 1) * 100
                    peak = np.maximum.accumulate(eq); maxdd = float(((eq - peak) / peak).min() * 100)
                    cells.append({"cadence": cad, "asset": sym, "config": name, "family": fam_of[name],
                                  "speed": SPEED(name), "exit": ex, "comb": comb, "jan": jan, "feb": feb,
                                  "maxdd": maxdd, "fip": float((eq > 1).mean()) * 100, "trades": ntr})
                    # keep one representative equity curve per (cadence, exit) for the chart: best 2MA-slow
                    if fam_of[name] == "2MA" and SPEED(name) == "slow(60-150)":
                        key = (cad, ex)
                        if key not in curves or comb > curves[key][0]:
                            curves[key] = (comb, eq.copy(), pd.to_datetime(ms[wm], unit="ms"))
    return cells, curves


def main() -> int:
    cells, curves = run()
    print(f"## COMBINED Jan+Feb 2020 -- {len(cells)} cells (24 cfg/fam x 7 assets x {len(TFS)} cad x {len(EXITS)} exits)")
    print(f"   window {START}..{END}, split {SPLIT} (Jan rally | Feb top+reversal)\n")

    # A. combined ROI by cadence x exit (+ Jan/Feb split, maxDD)
    print("[A] combined ROI by CADENCE x EXIT  (comb% = jan% then feb%; maxDD)")
    print(f"   {'cadence':8} {'exit':10} {'comb%':>7} {'jan%':>7} {'feb%':>7} {'maxDD%':>8} {'trades':>7} {'%pos':>5}")
    for cad in TFS:
        for ex in EXITS:
            v = [x for x in cells if x["cadence"] == cad and x["exit"] == ex]
            if not v: continue
            cm = np.array([x["comb"] for x in v])
            print(f"   {cad:8} {ex:10} {cm.mean():>7.1f} {np.mean([x['jan'] for x in v]):>7.1f} "
                  f"{np.mean([x['feb'] for x in v]):>7.1f} {np.mean([x['maxdd'] for x in v]):>8.1f} "
                  f"{np.mean([x['trades'] for x in v]):>7.0f} {100*(cm>0).mean():>4.0f}%")
        print()

    # B. EXIT mechanism ranking over the full span (the reversal test)
    print("[B] EXIT mechanism over the COMBINED span (pooled all cadences) -- who keeps the gains?")
    print(f"   {'exit':10} {'comb%':>7} {'jan%':>7} {'feb%':>7} {'gave-back':>10} {'maxDD%':>8} {'%pos':>5}")
    for ex in EXITS:
        v = [x for x in cells if x["exit"] == ex]
        cm = np.array([x["comb"] for x in v]); jan = np.mean([x["jan"] for x in v]); feb = np.mean([x["feb"] for x in v])
        gb = feb  # feb-half return; negative = gave back gains in the reversal
        print(f"   {ex:10} {cm.mean():>7.1f} {jan:>7.1f} {feb:>7.1f} {gb:>+9.1f}% {np.mean([x['maxdd'] for x in v]):>8.1f} {100*(cm>0).mean():>4.0f}%")

    # C. family x speed: combined + Feb robustness
    print("\n[C] FAMILY x SPEED -- combined ROI + Feb-half (robustness through the reversal), signalflip")
    print(f"   {'family':6} {'speed':12} {'comb%':>7} {'jan%':>7} {'feb%':>7} {'maxDD%':>8} {'%pos':>5}")
    for fam in ("2MA", "3MA"):
        for sp in BUCKETS:
            v = [x for x in cells if x["exit"] == "signalflip" and x["family"] == fam and x["speed"] == sp]
            if not v: continue
            cm = np.array([x["comb"] for x in v])
            print(f"   {fam:6} {sp:12} {cm.mean():>7.1f} {np.mean([x['jan'] for x in v]):>7.1f} "
                  f"{np.mean([x['feb'] for x in v]):>7.1f} {np.mean([x['maxdd'] for x in v]):>8.1f} {100*(cm>0).mean():>4.0f}%")

    # D. best combined config per cadence (signalflip)
    print("\n[D] BEST combined config per cadence (by combined ROI, signalflip):")
    for cad in TFS:
        v = [x for x in cells if x["cadence"] == cad and x["exit"] == "signalflip"]
        byc = {}
        for x in v: byc.setdefault(x["config"], []).append(x["comb"])
        best = max(byc.items(), key=lambda kv: np.mean(kv[1]))
        print(f"   {cad:5} {best[0]:16} comb {np.mean(best[1]):+6.1f}%  ({SPEED(best[0])})")

    # ---- chart: combined equity (best 2MA-slow) per cadence + exit comparison ----
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    for cad in TFS:
        key = (cad, "signalflip")
        if key in curves:
            _, eq, dts = curves[key]
            ax[0].plot(dts, eq, label=f"{cad} ({curves[key][0]:+.0f}%)", lw=1.4)
    ax[0].axvline(pd.Timestamp(SPLIT), color="k", ls="--", lw=0.8); ax[0].axhline(1.0, color="grey", ls=":", lw=0.6)
    ax[0].annotate("Jan | Feb", (pd.Timestamp(SPLIT), ax[0].get_ylim()[1]), fontsize=8)
    ax[0].set_ylabel("$1 equity (best 2MA-slow, signalflip)"); ax[0].legend(fontsize=8)
    ax[0].set_title("Combined equity per cadence (rally -> Feb top+reversal)")
    x = np.arange(len(TFS)); w = 0.2
    for i, ex in enumerate(EXITS):
        vals = [np.mean([c["comb"] for c in cells if c["cadence"] == cad and c["exit"] == ex] or [np.nan]) for cad in TFS]
        ax[1].bar(x + (i - 1.5) * w, vals, w, label=ex)
    ax[1].set_xticks(x); ax[1].set_xticklabels(TFS); ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_ylabel("combined ROI %"); ax[1].legend(fontsize=8)
    ax[1].set_title("Exit mechanism x cadence -- combined 2-month ROI")
    fig.tight_layout(); out = OUT / "charts" / "combined_jan_feb.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
